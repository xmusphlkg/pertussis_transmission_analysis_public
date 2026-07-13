## Extended Data Figure 5 data preparation ------------------------------------

extended_data_figure_5_vaccine_short_labels <- function() {
  c(
    no_vaccine = "No vaccine",
    symptom_protective = "Current aP",
    infection_blocking = "Inf.-blocking",
    transmission_blocking = "Trans.-blocking",
    next_generation = "Upper-bound"
  )
}

extended_data_figure_5_metric_short_labels <- function() {
  c(
    relative_reduction_infant_cases = "Infant",
    relative_reduction_reported_cases = "Reported",
    relative_reduction_total_infections = "All",
    relative_reduction_resistant_infections = "Resistant"
  )
}

extended_data_figure_5_origin_columns <- function() {
  c(
    maternal_origin_infection_share = "Maternal",
    dose1_origin_infection_share = "Dose 1",
    dose2_origin_infection_share = "Dose 2",
    dose3plus_origin_infection_share = "Dose 3+",
    waned_origin_infection_share = "Waned"
  )
}

load_extended_data_figure_5_scenario_table <- function() {
  readr::read_csv(model_path("manuscript_notes", "scenario_table.csv"), show_col_types = FALSE) %>%
    mutate(
      scenario = factor(scenario, levels = vaccine_levels),
      scenario_label = factor(vaccine_labels[as.character(scenario)], levels = vaccine_labels[vaccine_levels])
    )
}

load_extended_data_figure_5_baseline <- function() {
  read_model_table(model_path("outputs", "summaries", "country_scenarios_summary")) %>%
    add_country_label()
}

load_extended_data_figure_5_vaccine_summary <- function() {
  read_model_table(model_path("outputs", "summaries", "vaccine_scenarios_summary")) %>%
    add_country_label() %>%
    mutate(
      scenario = factor(scenario, levels = vaccine_levels),
      scenario_label = factor(vaccine_labels[as.character(scenario)], levels = vaccine_labels[vaccine_levels])
    )
}

load_extended_data_figure_5_vaccine_sim <- function() {
  read_model_table_optional(model_path("outputs", "simulations", "vaccine_scenarios"))
}

load_extended_data_figure_5_inputs <- function() {
  list(
    scenario_table = load_extended_data_figure_5_scenario_table(),
    baseline = load_extended_data_figure_5_baseline(),
    vaccine_summary = load_extended_data_figure_5_vaccine_summary(),
    vaccine_sim = load_extended_data_figure_5_vaccine_sim()
  )
}

prepare_extended_data_figure_5_data <- function(inputs = load_extended_data_figure_5_inputs()) {
  vaccine_short_labels <- extended_data_figure_5_vaccine_short_labels()
  metric_short_labels <- extended_data_figure_5_metric_short_labels()
  origin_columns <- extended_data_figure_5_origin_columns()

  vaccine_parameters <- inputs$scenario_table %>%
    select(scenario_label, VE_sus, VE_sym, VE_inf, VE_dur) %>%
    pivot_longer(-scenario_label, names_to = "effect", values_to = "value") %>%
    mutate(effect = factor(effect, levels = c("VE_sus", "VE_sym", "VE_inf", "VE_dur")))

  vaccine_reduction_data <- inputs$vaccine_summary %>%
    filter(scenario != "no_vaccine") %>%
    mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels[-1]])) %>%
    select(
      country_label,
      scenario_short,
      relative_reduction_infant_cases,
      relative_reduction_total_infections,
      relative_reduction_reported_cases,
      relative_reduction_resistant_infections
    ) %>%
    pivot_longer(-c(country_label, scenario_short), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric_short_labels[metric], levels = c("Infant", "Reported", "All", "Resistant")))

  origin_share_data <- inputs$vaccine_summary %>%
    mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels])) %>%
    select(scenario_short, all_of(names(origin_columns))) %>%
    pivot_longer(-scenario_short, names_to = "origin", values_to = "share") %>%
    mutate(origin = factor(origin_columns[origin], levels = origin_columns)) %>%
    group_by(scenario_short, origin) %>%
    summarise(
      median_share = median(share, na.rm = TRUE),
      q025 = interval_quantile(share, 0.025),
      q975 = interval_quantile(share, 0.975),
      .groups = "drop"
    ) %>%
    mutate(
      interval_text = interval_label(median_share, q025, q975, formatter = label_lancet_percent(accuracy = 1)),
      text_colour = if_else(median_share < 0.08, "white", "black")
    )

  vaccine_sim <- inputs$vaccine_sim
  require_efigure_timeseries(vaccine_sim, "eFigure 5 representative vaccine trajectories", "vaccine_scenarios")

  if (nrow(vaccine_sim) > 0) {
    vaccine_ts <- vaccine_sim %>%
      filter(country %in% c("Australia", "China")) %>%
      add_country_label() %>%
      left_join(inputs$baseline %>% select(country, infant_population), by = "country") %>%
      mutate(
        scenario = factor(scenario, levels = vaccine_levels),
        scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels])
      ) %>%
      group_by(country_label, scenario_short, time) %>%
      summarise(
        simulation_year = first(time) / 365,
        infant_population = max(infant_population, na.rm = TRUE),
        infant_case_rate = sum(infant_case_rate_per_day, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(infant_case_incidence = infant_case_rate / pmax(infant_population, 1e-9) * 365 * 1e5)
  } else {
    vaccine_ts <- inputs$vaccine_summary %>%
      filter(country %in% c("Australia", "China")) %>%
      mutate(scenario_short = factor(vaccine_short_labels[as.character(scenario)], levels = vaccine_short_labels[vaccine_levels]))
  }

  list(
    scenario_table = inputs$scenario_table,
    baseline = inputs$baseline,
    vaccine_summary = inputs$vaccine_summary,
    vaccine_sim = vaccine_sim,
    vaccine_parameters = vaccine_parameters,
    vaccine_reduction_data = vaccine_reduction_data,
    origin_share_data = origin_share_data,
    vaccine_ts = vaccine_ts,
    has_vaccine_sim = nrow(vaccine_sim) > 0
  )
}
