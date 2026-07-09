## Extended Data Figure 6 data preparation ------------------------------------

extended_data_figure_6_input_paths <- function() {
  list(
    resistance_summary = model_path("outputs", "summaries", "resistance_scenarios_summary"),
    resistance_targets = model_path("publication_inputs", "resistance_scenario_table.csv"),
    resistance_sim = model_path("outputs", "simulations", "resistance_scenarios")
  )
}

load_extended_data_figure_6_resistance_summary <- function(path = extended_data_figure_6_input_paths()$resistance_summary) {
  read_model_table(path) %>%
    add_country_label() %>%
    mutate(
      scenario = factor(scenario, levels = resistance_levels),
      scenario_label = factor(
        resistance_labels[as.character(scenario)],
        levels = resistance_labels[resistance_levels]
      )
    )
}

load_extended_data_figure_6_resistance_targets <- function(path = extended_data_figure_6_input_paths()$resistance_targets) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    transmute(
      scenario = as.character(scenario),
      scenario_target = as.numeric(target_prevalence_at_analysis_start)
    )
}

load_extended_data_figure_6_resistance_sim <- function(path = extended_data_figure_6_input_paths()$resistance_sim) {
  read_model_table_optional(path)
}

load_extended_data_figure_6_inputs <- function(paths = extended_data_figure_6_input_paths()) {
  list(
    resistance_summary = load_extended_data_figure_6_resistance_summary(paths$resistance_summary),
    resistance_targets = load_extended_data_figure_6_resistance_targets(paths$resistance_targets),
    resistance_sim = load_extended_data_figure_6_resistance_sim(paths$resistance_sim)
  )
}

prepare_extended_data_figure_6_data <- function(inputs = load_extended_data_figure_6_inputs()) {
  resistance_summary <- inputs$resistance_summary
  resistance_targets <- inputs$resistance_targets
  resistance_sim <- inputs$resistance_sim

  require_efigure_timeseries(
    resistance_sim,
    "eFigure 6 sensitive and resistant strain trajectories",
    "resistance_scenarios"
  )

  realized_start_diagnostic <- resistance_summary %>%
    mutate(scenario_key = as.character(scenario)) %>%
    left_join(resistance_targets, by = c("scenario_key" = "scenario")) %>%
    mutate(
      target_resistance = if_else(
        scenario_key == "country_timeline",
        resistant_fraction_start,
        scenario_target
      )
    )

  resistant_rate <- resistance_summary %>%
    filter(!is.na(scenario_label)) %>%
    mutate(
      resistant_rate = resistant_infections / pmax(total_population * analysis_years, 1e-9) * 1e5
    )

  treatment_pep <- resistance_summary %>%
    mutate(
      `Treated cases` = treated_cases / pmax(total_population * analysis_years, 1e-9) * 1e5,
      `PEP-averted cases` = PEP_averted_cases / pmax(total_population * analysis_years, 1e-9) * 1e5
    ) %>%
    select(scenario_label, `Treated cases`, `PEP-averted cases`) %>%
    pivot_longer(-scenario_label, names_to = "metric", values_to = "rate") %>%
    filter(!is.na(scenario_label)) %>%
    group_by(scenario_label, metric) %>%
    summarise(
      median_rate = median(rate, na.rm = TRUE),
      q025 = interval_quantile(rate, 0.025),
      q975 = interval_quantile(rate, 0.975),
      q25 = interval_quantile(rate, 0.25),
      q75 = interval_quantile(rate, 0.75),
      .groups = "drop"
    )

  if (nrow(resistance_sim) > 0) {
    resistance_ts <- resistance_sim %>%
      filter(country %in% c("Australia", "China"), scenario == "country_timeline") %>%
      add_country_label() %>%
      mutate(strain_label = factor(str_to_title(strain), levels = c("Sensitive", "Resistant"))) %>%
      group_by(country_label, strain_label, time) %>%
      summarise(
        simulation_year = first(time) / 365,
        total_population = max(total_population, na.rm = TRUE),
        infection_rate = sum(total_infection_rate_per_day, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(infection_incidence = infection_rate / pmax(total_population, 1e-9) * 365 * 1e5)

    resistance_ts_mode <- "simulation"
  } else {
    resistance_ts <- resistance_summary %>%
      filter(country %in% c("Australia", "China"), as.character(scenario) == "country_timeline") %>%
      mutate(
        `Sensitive` = pmax(total_infections - resistant_infections, 0) /
          pmax(total_population * analysis_years, 1e-9) * 1e5,
        `Resistant` = resistant_infections /
          pmax(total_population * analysis_years, 1e-9) * 1e5
      ) %>%
      select(country_label, Sensitive, Resistant) %>%
      pivot_longer(c(Sensitive, Resistant), names_to = "strain_label", values_to = "infection_incidence") %>%
      mutate(strain_label = factor(strain_label, levels = c("Sensitive", "Resistant")))

    resistance_ts_mode <- "summary_fallback"
  }

  list(
    resistance_summary = resistance_summary,
    resistance_targets = resistance_targets,
    resistance_sim = resistance_sim,
    realized_start_diagnostic = realized_start_diagnostic,
    resistant_rate = resistant_rate,
    treatment_pep = treatment_pep,
    resistance_ts = resistance_ts,
    resistance_ts_mode = resistance_ts_mode
  )
}
