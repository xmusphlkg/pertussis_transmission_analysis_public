## Extended Data Figure 7 data preparation ------------------------------------

extended_data_figure_7_intervention_labels <- function() {
  c(
    higher_child_coverage = "Nominal\ncoverage",
    adolescent_booster = "Adolescent\nbooster",
    pregnancy_tdap_scaleup = "Pregnancy\nTdap",
    cocooning_adjunct = "Close-contact\nadjunct",
    maternal_immunization = "Infant-exposure\nstrategy",
    targeted_pep_high_risk = "Targeted\nPEP",
    resistance_guided_treatment = "Resistance-guided\nmanagement",
    next_generation_vaccine = "High-blocking\nvaccine target",
    combined_strategy = "Combined\nstrategy"
  )
}

extended_data_figure_7_metric_short_labels <- function() {
  c(
    relative_reduction_infant_cases = "Infant",
    relative_reduction_reported_cases = "Reported",
    relative_reduction_total_infections = "All",
    relative_reduction_resistant_infections = "Resistant"
  )
}

load_extended_data_figure_7_intervention_summary <- function() {
  read_model_table(model_path("outputs", "summaries", "intervention_scenarios_summary")) %>%
    add_country_label() %>%
    mutate(
      scenario = factor(scenario, levels = c("current", intervention_levels)),
      scenario_label = factor(
        c(current = "Current", intervention_labels)[as.character(scenario)],
        levels = c("Current", intervention_labels[intervention_levels])
      )
    )
}

load_extended_data_figure_7_inputs <- function() {
  list(
    intervention_summary = load_extended_data_figure_7_intervention_summary(),
    maternal_decomposition_summary = read_model_table_optional(
      model_path("outputs", "summaries", "maternal_decomposition_summary")
    )
  )
}

prepare_extended_data_figure_7_data <- function(inputs = load_extended_data_figure_7_inputs()) {
  efig7_intervention_labels <- extended_data_figure_7_intervention_labels()
  metric_short_labels <- extended_data_figure_7_metric_short_labels()

  intervention_effects <- inputs$intervention_summary %>%
    mutate(scenario_key = as.character(scenario)) %>%
    filter(scenario_key %in% intervention_levels) %>%
    mutate(
      scenario = factor(scenario_key, levels = intervention_levels),
      scenario_label = factor(efig7_intervention_labels[scenario_key], levels = efig7_intervention_labels[intervention_levels]),
      scenario_short = factor(efig7_intervention_labels[scenario_key], levels = efig7_intervention_labels[intervention_levels])
    )

  intervention_levers <- tribble(
    ~scenario, ~lever,
    "higher_child_coverage", "Nominal\ncoverage",
    "adolescent_booster", "Adolescent\nbooster",
    "pregnancy_tdap_scaleup", "Pregnancy\nTdap",
    "cocooning_adjunct", "Close-contact\nadjunct",
    "maternal_immunization", "Pregnancy\nTdap",
    "maternal_immunization", "Close-contact\nadjunct",
    "targeted_pep_high_risk", "Targeted\nPEP",
    "resistance_guided_treatment", "Resistance-guided\nmanagement",
    "next_generation_vaccine", "High-blocking\nvaccine target",
    "combined_strategy", "Pregnancy\nTdap",
    "combined_strategy", "Close-contact\nadjunct",
    "combined_strategy", "Adolescent\nbooster",
    "combined_strategy", "Targeted\nPEP",
    "combined_strategy", "Resistance-guided\nmanagement",
    "combined_strategy", "Transmission-blocking\nvaccine"
  ) %>%
    mutate(active = TRUE)

  lever_levels <- c(
    "Nominal\ncoverage", "Adolescent\nbooster", "Pregnancy\nTdap",
    "Close-contact\nadjunct", "Targeted\nPEP", "Resistance-guided\nmanagement",
    "High-blocking\nvaccine target", "Transmission-blocking\nvaccine"
  )

  lever_matrix <- expand_grid(
    scenario = intervention_levels,
    lever = lever_levels
  ) %>%
    left_join(intervention_levers, by = c("scenario", "lever")) %>%
    mutate(
      active = replace_na(active, FALSE),
      scenario_label = factor(efig7_intervention_labels[scenario], levels = rev(efig7_intervention_labels[intervention_levels])),
      lever = factor(lever, levels = lever_levels)
    )

  intervention_outcomes <- intervention_effects %>%
    select(
      country_label,
      scenario_short,
      relative_reduction_infant_cases,
      relative_reduction_reported_cases,
      relative_reduction_total_infections,
      relative_reduction_resistant_infections
    ) %>%
    pivot_longer(-c(country_label, scenario_short), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric_short_labels[metric], levels = c("Infant", "Reported", "All", "Resistant")))

  maternal_decomp_levels <- c(
    "pregnancy_tdap_scaleup", "maternal_adult_boosting_only",
    "maternal_cocooning_only", "maternal_immunization"
  )
  maternal_decomp_labels <- c(
    pregnancy_tdap_scaleup = "Pregnancy Tdap",
    maternal_adult_boosting_only = "Adult boosting",
    maternal_cocooning_only = "Contact reduction",
    maternal_immunization = "Infant-exposure strategy"
  )
  maternal_decomp_colours <- c(
    "Pregnancy Tdap" = manuscript_colour("sky"),
    "Adult boosting" = manuscript_colour("orange"),
    "Contact reduction" = manuscript_colour("green"),
    "Infant-exposure strategy" = manuscript_colour("magenta")
  )

  maternal_decomposition_components <- inputs$intervention_summary %>%
    mutate(scenario_key = if ("intervention" %in% names(.)) as.character(intervention) else as.character(scenario)) %>%
    filter(scenario_key %in% maternal_decomp_levels) %>%
    transmute(country_label, scenario = scenario_key, relative_reduction_infant_cases)

  if (nrow(maternal_decomposition_components) == 0 && nrow(inputs$maternal_decomposition_summary) > 0) {
    maternal_current <- inputs$intervention_summary %>%
      mutate(scenario_key = if ("intervention" %in% names(.)) as.character(intervention) else as.character(scenario)) %>%
      filter(scenario_key == "current") %>%
      select(country, current_infant = annualized_infant_cases_per_100k)

    maternal_decomposition_components <- inputs$maternal_decomposition_summary %>%
      add_country_label() %>%
      left_join(maternal_current, by = "country") %>%
      transmute(
        country_label,
        scenario,
        relative_reduction_infant_cases = (current_infant - annualized_infant_cases_per_100k) /
          pmax(current_infant, 1e-9)
      )
  }

  maternal_decomp <- maternal_decomposition_components %>%
    filter(scenario %in% maternal_decomp_levels) %>%
    mutate(
      component = factor(
        maternal_decomp_labels[as.character(scenario)],
        levels = maternal_decomp_labels[maternal_decomp_levels]
      )
    )

  missing_decomp_components <- setdiff(maternal_decomp_levels, unique(as.character(maternal_decomp$scenario)))
  expected_decomp_cells <- expand_grid(
    scenario = maternal_decomp_levels,
    country_label = country_label_levels
  )
  observed_decomp_cells <- maternal_decomp %>%
    transmute(
      scenario = as.character(scenario),
      country_label = as.character(country_label)
    ) %>%
    distinct()
  missing_decomp_cells <- expected_decomp_cells %>%
    anti_join(observed_decomp_cells, by = c("scenario", "country_label"))
  if (length(missing_decomp_components) > 0 || nrow(missing_decomp_cells) > 0) {
    stop(
      "eFigure 7 panel C requires complete infant-exposure decomposition source data. Missing components: ",
      paste(missing_decomp_components, collapse = ", "),
      "; missing component-country cells: ",
      paste(
        paste0(missing_decomp_cells$scenario, "/", missing_decomp_cells$country_label),
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  maternal_decomp_agg <- maternal_decomp %>%
    group_by(component) %>%
    summarise(
      median_reduction = median(relative_reduction_infant_cases, na.rm = TRUE),
      q025 = interval_quantile(relative_reduction_infant_cases, 0.025),
      q975 = interval_quantile(relative_reduction_infant_cases, 0.975),
      q25 = interval_quantile(relative_reduction_infant_cases, 0.25),
      q75 = interval_quantile(relative_reduction_infant_cases, 0.75),
      .groups = "drop"
    )

  list(
    intervention_effects = intervention_effects,
    lever_matrix = lever_matrix,
    intervention_outcomes = intervention_outcomes,
    maternal_decomp = maternal_decomp,
    maternal_decomp_agg = maternal_decomp_agg,
    maternal_decomp_colours = maternal_decomp_colours
  )
}
