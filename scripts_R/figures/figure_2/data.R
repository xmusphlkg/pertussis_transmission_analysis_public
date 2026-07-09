## Figure 2 data preparation ---------------------------------------------------

figure_2_strategy_order <- function() {
  c(
    "maternal_immunization",
    "timeliness_only",
    "cocooning_adjunct",
    "targeted_pep_high_risk",
    "pregnancy_tdap_scaleup",
    "adolescent_booster"
  )
}

figure_2_winning_strategy_group_order <- function() {
  c(
    "timeliness_only",
    "maternal_immunization",
    "targeted_pep_high_risk"
  )
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

figure_2_selected_strategy_labels <- function() {
  c(
    higher_child_coverage = "Coverage",
    timeliness_only = "Routine schedule\ntimeliness",
    maternal_immunization = "Infant-exposure\npackage",
    cocooning_adjunct = "Close-contact\nadult adjuncts",
    targeted_pep_high_risk = "Targeted high-risk\nPEP",
    adolescent_booster = "Adolescent booster\nscale-up",
    pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
  )
}

load_figure_2_inputs <- function() {
  list(
    frontier = read_table("lancet_child_adolescent_decision_frontier.csv"),
    primary_interval_audit = read_table("figure2c_programme_primary_predictive_interval_audit.csv")
  )
}

prepare_figure_2_data <- function(inputs = load_figure_2_inputs()) {
  strategy_order <- figure_2_strategy_order()
  winning_strategy_group_order <- figure_2_winning_strategy_group_order()
  strategy_labels <- figure_2_strategy_labels()
  strategy_overview_labels <- figure_2_strategy_overview_labels()
  strategy_heatmap_labels <- figure_2_strategy_heatmap_labels()
  selected_strategy_labels <- figure_2_selected_strategy_labels()

  rank_levels <- 1:6
  rank_labels <- c("1st", "2nd", "3rd", "4th", "5th", "6th")
  rank_colours <- setNames(
    grDevices::colorRampPalette(manuscript_continuous_burden)(length(rank_labels)),
    rank_labels
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
      non_dominated_lancet_outcome = as.logical(non_dominated_lancet_outcome),
      primary_case_rank_within_constraint = as.numeric(primary_case_rank_within_constraint)
    )

  primary_interval_audit <- inputs$primary_interval_audit %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      strategy = as.character(scenario_key),
      primary_reduction_interval_q025 = as.numeric(reduction_q025),
      primary_reduction_interval_q975 = as.numeric(reduction_q975),
      current_rate_interval_q025 = as.numeric(current_rate_q025),
      current_rate_interval_q975 = as.numeric(current_rate_q975),
      intervention_rate_interval_q025 = as.numeric(intervention_rate_q025),
      intervention_rate_interval_q975 = as.numeric(intervention_rate_q975)
    ) %>%
    select(
      country,
      strategy,
      primary_reduction_interval_q025,
      primary_reduction_interval_q975,
      current_rate_interval_q025,
      current_rate_interval_q975,
      intervention_rate_interval_q025,
      intervention_rate_interval_q975,
      interval_basis
    )

  country_order <- frontier_raw %>%
    filter(strategy == "current", optimization_constraint == "program_only") %>%
    arrange(desc(current_primary_cases_per_100k)) %>%
    pull(country_label_text)

  program_frontier <- frontier_raw %>%
    filter(optimization_constraint == "program_only", strategy %in% strategy_order) %>%
    mutate(
      country_label = factor(country_label_text, levels = rev(country_order)),
      strategy_factor = factor(strategy, levels = strategy_order),
      strategy_label_plot = factor(strategy_labels[strategy], levels = rev(strategy_labels[strategy_order])),
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

  heatmap_group_separators <- length(heatmap_country_order) -
    cumsum(heatmap_group_sizes$profiles)[-nrow(heatmap_group_sizes)] + 0.5

  selected_program <- selected_program %>%
    mutate(
      country_label_heatmap = factor(country_label_text, levels = rev(heatmap_country_order)),
      country_label_margin = factor(country_label_text, levels = rev(heatmap_country_order))
    )

  selected_strategy_order <- strategy_order[strategy_order %in% selected_program$winning_strategy]
  selected_strategy_legend_labels <- stringr::str_replace_all(
    selected_strategy_labels[selected_strategy_order],
    "\n",
    " "
  )

  strategy_distribution <- program_frontier %>%
    group_by(strategy, strategy_label, strategy_overview_plot) %>%
    summarise(
      median_reduction = median(primary_case_reduction, na.rm = TRUE),
      q25_reduction = interval_quantile(primary_case_reduction, 0.25),
      q75_reduction = interval_quantile(primary_case_reduction, 0.75),
      min_reduction = min(primary_case_reduction, na.rm = TRUE),
      max_reduction = max(primary_case_reduction, na.rm = TRUE),
      countries_ranked_first = sum(decision_rank == 1, na.rm = TRUE),
      countries_non_dominated = sum(non_dominated_lancet_outcome, na.rm = TRUE),
      .groups = "drop"
    )

  program_heatmap <- program_frontier %>%
    left_join(primary_interval_audit, by = c("country", "strategy")) %>%
    mutate(
      country_label = factor(country_label_text, levels = rev(heatmap_country_order)),
      strategy_axis = factor(strategy, levels = strategy_order),
      interval_label = if_else(
        is.finite(primary_reduction_interval_q025) & is.finite(primary_reduction_interval_q975),
        paste0(
          "[",
          lancet_percent(primary_reduction_interval_q025, accuracy = 0.1),
          ", ",
          lancet_percent(primary_reduction_interval_q975, accuracy = 0.1),
          "]"
        ),
        NA_character_
      ),
      effect_label = if_else(
        is.na(interval_label),
        lancet_percent(primary_case_reduction, accuracy = 1),
        paste0(lancet_percent(primary_case_reduction, accuracy = 1), "\n", interval_label)
      ),
      effect_text_colour = if_else(primary_case_reduction >= 0.28, "white", lancet_text_colour),
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
    winning_strategy_group_order = winning_strategy_group_order,
    strategy_labels = strategy_labels,
    strategy_overview_labels = strategy_overview_labels,
    strategy_heatmap_labels = strategy_heatmap_labels,
    selected_strategy_labels = selected_strategy_labels,
    rank_levels = rank_levels,
    rank_labels = rank_labels,
    rank_colours = rank_colours,
    frontier_raw = frontier_raw,
    primary_interval_audit = primary_interval_audit,
    country_order = country_order,
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
