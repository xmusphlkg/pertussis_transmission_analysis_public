## Figure 3 data preparation ---------------------------------------------------

figure_3_programme_strategies <- function() {
  c(
    "maternal_immunization",
    "timeliness_only",
    "pregnancy_tdap_scaleup",
    "cocooning_adjunct",
    "targeted_pep_high_risk",
    "adolescent_booster"
  )
}

figure_3_strategy_source_labels <- function() {
  c(
    higher_child_coverage = "Coverage",
    timeliness_only = "Routine schedule timeliness",
    maternal_immunization = "Infant-exposure package",
    cocooning_adjunct = "Close-contact adult adjunct",
    targeted_pep_high_risk = "Targeted high-risk PEP",
    adolescent_booster = "Adolescent booster scale-up",
    pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up"
  )
}

figure_3_strategy_plot_labels <- function() {
  c(
    higher_child_coverage = "Coverage",
    timeliness_only = "Routine schedule\ntimeliness",
    maternal_immunization = "Infant-exposure\npackage",
    cocooning_adjunct = "Close-contact\nadult adjunct",
    targeted_pep_high_risk = "Targeted high-risk\nPEP",
    adolescent_booster = "Adolescent booster\nscale-up",
    pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up"
  )
}

figure_3_endpoint_levels <- function() {
  c(
    "Infant cases",
    "Infant hospitalisations",
    "Infant deaths",
    "Children cases",
    "Adolescent cases",
    "All <18 cases"
  )
}

figure_3_age_stratum_levels <- function() {
  c("Infant", "Children", "Adolescent")
}

figure_3_age_stratum_colours <- function() {
  c(
    "Infant" = palette_discrete_primary_9[[7]],
    "Children" = palette_discrete_primary_9[[4]],
    "Adolescent" = palette_discrete_primary_9[[8]]
  )
}

figure_3_age_stratum_display_labels <- function() {
  c(
    "Infant" = "Infants 0-11 m",
    "Children" = "Children 1-9 y",
    "Adolescent" = "Adolescents 10-17 y"
  )
}

figure_3_endpoint_axis_labels <- function() {
  c(
    `Infant cases` = "Infant cases",
    `Infant hospitalisations` = "Infant hospitalisations",
    `Infant deaths` = "Infant deaths",
    `Children cases` = "Children cases",
    `Adolescent cases` = "Adolescent cases",
    `All <18 cases` = "All <18 cases"
  )
}

load_figure_3_inputs <- function() {
  list(
    burden = read_table("lancet_child_adolescent_strategy_burden.csv"),
    population = read_table("calibration_all_countries.csv")
  )
}

prepare_figure_3_data <- function(inputs = load_figure_3_inputs()) {
  programme_strategies <- figure_3_programme_strategies()
  strategy_source_labels <- figure_3_strategy_source_labels()
  strategy_plot_labels <- figure_3_strategy_plot_labels()
  endpoint_levels <- figure_3_endpoint_levels()
  age_stratum_levels <- figure_3_age_stratum_levels()
  age_stratum_colours <- figure_3_age_stratum_colours()
  age_stratum_display_labels <- figure_3_age_stratum_display_labels()
  endpoint_axis_labels <- figure_3_endpoint_axis_labels()

  burden <- inputs$burden %>%
    required_columns(
      c(
        "country", "strategy", "primary_cases_per_100k",
        "infant_cases_per_100k", "infant_hospitalizations_per_100k",
        "infant_deaths_per_100k", "child_1_9_cases_per_100k",
        "adolescent_cases_per_100k", "primary_case_reduction",
        "relative_reduction_infant_hospitalizations",
        "relative_reduction_infant_deaths"
      ),
      "Lancet child/adolescent strategy burden"
    ) %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_code = unname(country_codes[country]),
      strategy = as.character(strategy),
      strategy_label = unname(strategy_source_labels[strategy]),
      strategy_label_plot = factor(
        unname(strategy_plot_labels[strategy]),
        levels = rev(strategy_plot_labels[programme_strategies])
      ),
      primary_cases_per_100k = as.numeric(primary_cases_per_100k),
      infant_cases_per_100k = as.numeric(infant_cases_per_100k),
      infant_hospitalizations_per_100k = as.numeric(infant_hospitalizations_per_100k),
      infant_deaths_per_100k = as.numeric(infant_deaths_per_100k),
      child_1_9_cases_per_100k = as.numeric(child_1_9_cases_per_100k),
      adolescent_cases_per_100k = as.numeric(adolescent_cases_per_100k),
      primary_case_reduction = as.numeric(primary_case_reduction),
      relative_reduction_infant_hospitalizations = as.numeric(relative_reduction_infant_hospitalizations),
      relative_reduction_infant_deaths = as.numeric(relative_reduction_infant_deaths)
    )

  current_burden <- burden %>%
    filter(strategy == "current") %>%
    select(
      country,
      current_primary_cases_per_100k = primary_cases_per_100k,
      current_infant_cases_per_100k = infant_cases_per_100k,
      current_infant_hospitalizations_per_100k = infant_hospitalizations_per_100k,
      current_infant_deaths_per_100k = infant_deaths_per_100k,
      current_child_1_9_cases_per_100k = child_1_9_cases_per_100k,
      current_adolescent_cases_per_100k = adolescent_cases_per_100k
    )

  if (
    nrow(current_burden) != 9L ||
      anyDuplicated(current_burden$country) > 0L ||
      any(!is.finite(as.matrix(select(current_burden, -country)))) ||
      any(as.matrix(select(current_burden, -country)) <= 0)
  ) {
    stop("Figure 3 requires one positive common-current comparator for each of nine profiles.", call. = FALSE)
  }

  burden <- burden %>%
    select(
      -any_of(c(
        "current_primary_cases_per_100k",
        "current_infant_cases_per_100k",
        "current_infant_hospitalizations_per_100k",
        "current_infant_deaths_per_100k",
        "current_child_1_9_cases_per_100k",
        "current_adolescent_cases_per_100k"
      ))
    ) %>%
    left_join(current_burden, by = "country") %>%
    mutate(
      all_under18_case_reduction = 1 - primary_cases_per_100k / current_primary_cases_per_100k,
      infant_case_reduction = 1 - infant_cases_per_100k / pmax(current_infant_cases_per_100k, 1e-9),
      infant_hospitalization_reduction = 1 - infant_hospitalizations_per_100k /
        pmax(current_infant_hospitalizations_per_100k, 1e-9),
      infant_death_reduction = 1 - infant_deaths_per_100k /
        pmax(current_infant_deaths_per_100k, 1e-9),
      child_1_9_case_reduction = 1 - child_1_9_cases_per_100k / pmax(current_child_1_9_cases_per_100k, 1e-9),
      adolescent_case_reduction = 1 - adolescent_cases_per_100k / pmax(current_adolescent_cases_per_100k, 1e-9),
      stored_primary_comparator_drift_pp = 100 * (primary_case_reduction - all_under18_case_reduction),
      stored_hospitalization_comparator_drift_pp = 100 *
        (relative_reduction_infant_hospitalizations - infant_hospitalization_reduction),
      stored_death_comparator_drift_pp = 100 *
        (relative_reduction_infant_deaths - infant_death_reduction)
    )

  programme_burden <- burden %>%
    filter(strategy %in% programme_strategies)

  available_country_labels <- current_burden %>%
    transmute(country_label_text = format_country(country)) %>%
    pull(country_label_text)
  country_order <- main_figure_country_order(available_country_labels)
  if (length(country_order) != length(main_figure_country_label_levels) ||
      !setequal(country_order, available_country_labels)) {
    stop("Figure 3 countries do not match the fixed main-figure order.", call. = FALSE)
  }

  programme_burden <- programme_burden %>%
    mutate(country_label = factor(country_label_text, levels = rev(country_order)))

  endpoint_country <- programme_burden %>%
    transmute(
      country,
      country_label_text,
      country_code,
      strategy,
      strategy_label,
      strategy_label_plot,
      `Infant cases` = infant_case_reduction,
      `Infant hospitalisations` = infant_hospitalization_reduction,
      `Infant deaths` = infant_death_reduction,
      `Children cases` = child_1_9_case_reduction,
      `Adolescent cases` = adolescent_case_reduction,
      `All <18 cases` = all_under18_case_reduction
    ) %>%
    pivot_longer(
      cols = all_of(endpoint_levels),
      names_to = "endpoint",
      values_to = "relative_case_reduction"
    ) %>%
    mutate(endpoint = factor(endpoint, levels = endpoint_levels))

  endpoint_effect_matrix <- endpoint_country %>%
    group_by(strategy, strategy_label, strategy_label_plot, endpoint) %>%
    summarise(
      country_count = n(),
      median_relative_case_reduction = median(relative_case_reduction, na.rm = TRUE),
      q25_relative_case_reduction = interval_quantile(relative_case_reduction, 0.25),
      q75_relative_case_reduction = interval_quantile(relative_case_reduction, 0.75),
      min_relative_case_reduction = min(relative_case_reduction, na.rm = TRUE),
      max_relative_case_reduction = max(relative_case_reduction, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      endpoint = factor(endpoint, levels = endpoint_levels),
      interval_label = if_else(
        is.finite(q25_relative_case_reduction) & is.finite(q75_relative_case_reduction),
        paste0(
          "[",
          lancet_percent(q25_relative_case_reduction, accuracy = 0.1),
          ", ",
          lancet_percent(q75_relative_case_reduction, accuracy = 0.1),
          "]"
        ),
        NA_character_
      ),
      effect_label = if_else(
        is.na(interval_label),
        lancet_percent(median_relative_case_reduction, accuracy = 0.1),
        paste0(lancet_percent(median_relative_case_reduction, accuracy = 0.1), "\n", interval_label)
      ),
      effect_text_colour = if_else(median_relative_case_reduction >= 0.28, "white", lancet_text_colour)
    )

  endpoint_gap <- programme_burden %>%
    mutate(
      infant_minus_child_adolescent_gap_pp = 100 * (infant_case_reduction - all_under18_case_reduction)
    )

  endpoint_gap_summary <- endpoint_gap %>%
    group_by(strategy, strategy_label, strategy_label_plot) %>%
    summarise(
      country_count = n(),
      median_gap_pp = median(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
      q25_gap_pp = interval_quantile(infant_minus_child_adolescent_gap_pp, 0.25),
      q75_gap_pp = interval_quantile(infant_minus_child_adolescent_gap_pp, 0.75),
      min_gap_pp = min(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
      max_gap_pp = max(infant_minus_child_adolescent_gap_pp, na.rm = TRUE),
      .groups = "drop"
    )

  population <- inputs$population %>%
    mutate(country = stringr::str_replace_all(country, " ", "_")) %>%
    transmute(
      country,
      infant_population = as.numeric(infant_population),
      child_1_9_population = as.numeric(child_1_9_population),
      adolescent_population = as.numeric(adolescent_population),
      child_adolescent_population = as.numeric(child_adolescent_population)
    )

  age_contribution_country <- programme_burden %>%
    left_join(population, by = "country") %>%
    mutate(
      infant_averted_per_100k_under18 =
        (current_infant_cases_per_100k - infant_cases_per_100k) *
          infant_population / pmax(child_adolescent_population, 1e-9),
      child_1_9_averted_per_100k_under18 =
        (current_child_1_9_cases_per_100k - child_1_9_cases_per_100k) *
          child_1_9_population / pmax(child_adolescent_population, 1e-9),
      adolescent_averted_per_100k_under18 =
        (current_adolescent_cases_per_100k - adolescent_cases_per_100k) *
          adolescent_population / pmax(child_adolescent_population, 1e-9),
      component_sum_averted_per_100k_under18 =
        infant_averted_per_100k_under18 +
          child_1_9_averted_per_100k_under18 +
          adolescent_averted_per_100k_under18,
      primary_averted_per_100k_under18 = current_primary_cases_per_100k - primary_cases_per_100k,
      component_sum_difference_per_100k =
        component_sum_averted_per_100k_under18 - primary_averted_per_100k_under18
    )

  age_contribution_long <- age_contribution_country %>%
    select(
      country,
      strategy,
      strategy_label,
      strategy_label_plot,
      primary_averted_per_100k_under18,
      component_sum_averted_per_100k_under18,
      component_sum_difference_per_100k,
      infant_averted_per_100k_under18,
      child_1_9_averted_per_100k_under18,
      adolescent_averted_per_100k_under18
    ) %>%
    pivot_longer(
      cols = c(
        infant_averted_per_100k_under18,
        child_1_9_averted_per_100k_under18,
        adolescent_averted_per_100k_under18
      ),
      names_to = "age_contribution",
      values_to = "cases_averted_per_100k_under18"
    ) %>%
    mutate(
      age_contribution = recode(
        age_contribution,
        infant_averted_per_100k_under18 = "Infant",
        child_1_9_averted_per_100k_under18 = "Children",
        adolescent_averted_per_100k_under18 = "Adolescent"
      ),
      age_contribution = factor(age_contribution, levels = age_stratum_levels)
    )

  age_contribution_summary <- age_contribution_long %>%
    group_by(strategy, strategy_label, strategy_label_plot, age_contribution) %>%
    summarise(
      country_count = n(),
      median_cases_averted_per_100k_under18 = median(cases_averted_per_100k_under18, na.rm = TRUE),
      q25_cases_averted_per_100k_under18 = interval_quantile(cases_averted_per_100k_under18, 0.25),
      q75_cases_averted_per_100k_under18 = interval_quantile(cases_averted_per_100k_under18, 0.75),
      .groups = "drop"
    ) %>%
    mutate(age_contribution = factor(age_contribution, levels = age_stratum_levels))

  p3c_extent <- age_contribution_summary %>%
    group_by(strategy, strategy_label_plot) %>%
    summarise(
      stacked_median = sum(median_cases_averted_per_100k_under18, na.rm = TRUE),
      .groups = "drop"
    )

  p3c_x_upper <- ceiling((max(p3c_extent$stacked_median, na.rm = TRUE) + 2) / 5) * 5

  booster_effect_profile <- programme_burden %>%
    filter(strategy == "adolescent_booster") %>%
    transmute(
      country,
      country_label_text,
      country_code,
      all_under18_symptomatic_case_reduction = all_under18_case_reduction,
      adolescent_case_reduction,
      all_under18_cases_per_100k = primary_cases_per_100k,
      adolescent_cases_per_100k,
      current_all_under18_cases_per_100k = current_primary_cases_per_100k,
      current_adolescent_cases_per_100k
    ) %>%
    arrange(desc(adolescent_case_reduction), country_label_text) %>%
    mutate(
      adolescent_effect_rank = row_number(),
      country_label_booster = factor(country_label_text, levels = rev(country_order)),
      highlight_profile = adolescent_effect_rank == 1L
    )

  booster_effect_long <- booster_effect_profile %>%
    select(
      country,
      country_label_text,
      country_code,
      country_label_booster,
      highlight_profile,
      all_under18_symptomatic_case_reduction,
      adolescent_case_reduction
    ) %>%
    pivot_longer(
      cols = c(all_under18_symptomatic_case_reduction, adolescent_case_reduction),
      names_to = "outcome",
      values_to = "case_reduction"
    ) %>%
    mutate(
      outcome_label = recode(
        outcome,
        all_under18_symptomatic_case_reduction = "All <18 symptomatic cases",
        adolescent_case_reduction = "Adolescent cases"
      ),
      outcome_label = factor(outcome_label, levels = c("All <18 symptomatic cases", "Adolescent cases"))
    )

  booster_highlight_labels <- booster_effect_long %>%
    filter(highlight_profile) %>%
    mutate(
      label = lancet_percent(case_reduction, accuracy = 0.1),
      label_nudge_x = if_else(outcome == "adolescent_case_reduction", 0.018, -0.018),
      label_hjust = if_else(outcome == "adolescent_case_reduction", 0, 1)
    )

  booster_x_range <- range(
    c(
      0,
      booster_effect_profile$all_under18_symptomatic_case_reduction,
      booster_effect_profile$adolescent_case_reduction
    ),
    na.rm = TRUE
  )
  booster_x_limits <- c(
    floor((booster_x_range[[1]] - 0.02) / 0.05) * 0.05,
    ceiling((booster_x_range[[2]] + 0.02) / 0.05) * 0.05
  )
  booster_x_breaks <- seq(booster_x_limits[[1]], booster_x_limits[[2]], by = 0.10)

  list(
    programme_strategies = programme_strategies,
    strategy_source_labels = strategy_source_labels,
    strategy_plot_labels = strategy_plot_labels,
    endpoint_levels = endpoint_levels,
    age_stratum_levels = age_stratum_levels,
    age_stratum_colours = age_stratum_colours,
    age_stratum_display_labels = age_stratum_display_labels,
    endpoint_axis_labels = endpoint_axis_labels,
    burden = burden,
    programme_burden = programme_burden,
    country_order = country_order,
    endpoint_country = endpoint_country,
    endpoint_effect_matrix = endpoint_effect_matrix,
    endpoint_gap = endpoint_gap,
    endpoint_gap_summary = endpoint_gap_summary,
    population = population,
    age_contribution_country = age_contribution_country,
    age_contribution_long = age_contribution_long,
    age_contribution_summary = age_contribution_summary,
    age_contribution_extent = p3c_extent,
    age_contribution_x_upper = p3c_x_upper,
    booster_effect_profile = booster_effect_profile,
    booster_effect_long = booster_effect_long,
    booster_highlight_labels = booster_highlight_labels,
    booster_x_limits = booster_x_limits,
    booster_x_breaks = booster_x_breaks
  )
}
