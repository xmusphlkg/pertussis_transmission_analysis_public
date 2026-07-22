## Extended Data Figure 4 data preparation ------------------------------------

extended_data_figure_4_input_paths <- function() {
  list(
    country_sim = model_path("outputs", "simulations", "country_scenarios"),
    baseline = model_path("outputs", "summaries", "country_scenarios_summary")
  )
}

load_extended_data_figure_4_inputs <- function(paths = extended_data_figure_4_input_paths()) {
  country_sim <- read_model_table_optional(paths$country_sim)
  baseline <- read_model_table(paths$baseline)

  require_efigure_timeseries(
    country_sim,
    "eFigure 4 baseline temporal dynamics",
    "country_scenarios"
  )

  list(
    country_sim = country_sim,
    baseline = baseline
  )
}

prepare_extended_data_figure_4_baseline <- function(baseline) {
  baseline <- baseline %>%
    add_country_label()

  baseline_order <- main_figure_country_order(as.character(baseline$country_label))

  baseline %>%
    mutate(country_burden_order = factor(as.character(country_label), levels = rev(baseline_order)))
}

prepare_extended_data_figure_4_timeseries <- function(country_sim, baseline) {
  country_ts <- country_sim %>%
    add_country_label() %>%
    left_join(baseline %>% select(country, infant_population), by = "country") %>%
    mutate(
      country_code = factor(country_codes[country], levels = country_codes[country_levels]),
      age_group = factor(age_group, levels = names(age_labels), labels = age_labels),
      strain_label = factor(str_to_title(strain), levels = c("Sensitive", "Resistant"))
    )

  weekly_country <- country_ts %>%
    group_by(country, country_label, country_code, time) %>%
    summarise(
      simulation_year = first(time) / 365,
      total_population = max(total_population, na.rm = TRUE),
      infant_population = max(infant_population, na.rm = TRUE),
      total_infection_rate = sum(total_infection_rate_per_day, na.rm = TRUE),
      infant_case_rate = sum(infant_case_rate_per_day, na.rm = TRUE),
      resistant_infection_rate = sum(if_else(strain == "resistant", total_infection_rate_per_day, 0), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      infection_incidence = total_infection_rate / pmax(total_population, 1e-9) * 365 * 1e5,
      infant_case_incidence = infant_case_rate / pmax(infant_population, 1e-9) * 365 * 1e5,
      resistant_fraction = resistant_infection_rate / pmax(total_infection_rate, 1e-12)
    )

  age_strain_contribution <- country_ts %>%
    group_by(country_label, age_group, strain_label) %>%
    summarise(infections = sum(total_infections, na.rm = TRUE), .groups = "drop") %>%
    group_by(country_label) %>%
    mutate(country_share = infections / pmax(sum(infections, na.rm = TRUE), 1e-9)) %>%
    ungroup()

  list(
    country_ts = country_ts,
    weekly_country = weekly_country,
    age_strain_contribution = age_strain_contribution
  )
}

prepare_extended_data_figure_4_fallback <- function(baseline) {
  baseline_snapshot <- baseline %>%
    mutate(country_code = factor(country_codes[country], levels = country_codes[country_levels]))

  origin_share_fallback <- baseline_snapshot %>%
    select(
      country_label,
      maternal_origin_infection_share,
      dose1_origin_infection_share,
      dose2_origin_infection_share,
      dose3plus_origin_infection_share,
      waned_origin_infection_share
    ) %>%
    pivot_longer(-country_label, names_to = "origin", values_to = "share") %>%
    mutate(origin = factor(
      origin,
      levels = c(
        "maternal_origin_infection_share",
        "dose1_origin_infection_share",
        "dose2_origin_infection_share",
        "dose3plus_origin_infection_share",
        "waned_origin_infection_share"
      ),
      labels = c("Maternal", "Dose 1", "Dose 2", "Dose 3+", "Waned")
    ))

  list(
    baseline_snapshot = baseline_snapshot,
    origin_share_fallback = origin_share_fallback
  )
}

prepare_extended_data_figure_4_data <- function(inputs = load_extended_data_figure_4_inputs()) {
  baseline <- prepare_extended_data_figure_4_baseline(inputs$baseline)
  has_timeseries <- nrow(inputs$country_sim) > 0

  branch_data <- if (has_timeseries) {
    prepare_extended_data_figure_4_timeseries(inputs$country_sim, baseline)
  } else {
    prepare_extended_data_figure_4_fallback(baseline)
  }

  c(
    list(
      country_sim = inputs$country_sim,
      baseline = baseline,
      has_timeseries = has_timeseries
    ),
    branch_data
  )
}
