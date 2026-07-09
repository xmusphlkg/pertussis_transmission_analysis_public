## Extended Data Figure 8 data preparation ------------------------------------

extended_data_figure_8_input_paths <- function() {
  list(
    hindcast = model_path("outputs", "tables", "resistance_hindcast_results.csv"),
    scores = model_path("outputs", "tables", "resistance_hindcast_scores.csv")
  )
}

load_extended_data_figure_8_inputs <- function(paths = extended_data_figure_8_input_paths()) {
  if (!file.exists(paths$hindcast)) {
    stop(
      "eFigure 8 requires outputs/tables/resistance_hindcast_results.csv. ",
      "Run python -m src_python.simulation.run_resistance_hindcast before plotting."
    )
  }
  if (!file.exists(paths$scores)) {
    stop(
      "eFigure 8 requires outputs/tables/resistance_hindcast_scores.csv. ",
      "Run python -m src_python.simulation.run_resistance_hindcast before plotting."
    )
  }

  list(
    hindcast = readr::read_csv(paths$hindcast, show_col_types = FALSE),
    scores = readr::read_csv(paths$scores, show_col_types = FALSE)
  )
}

prepare_extended_data_figure_8_data <- function(inputs = load_extended_data_figure_8_inputs()) {
  hindcast <- inputs$hindcast %>%
    mutate(fitness_label = sprintf("%.2f", fitness_R))

  observed_data <- hindcast %>%
    filter(!is.na(observed_fraction)) %>%
    select(country, calendar_year, observed_fraction, observed_lower, observed_upper) %>%
    distinct() %>%
    rename(lower = observed_lower, upper = observed_upper)

  scores <- inputs$scores %>%
    mutate(
      fitness_label = sprintf("%.2f", fitness_R),
      country_label = factor(country, levels = c("China", "Japan", "Australia"))
    )

  best_fitness <- scores %>%
    group_by(country) %>%
    filter(mean_absolute_error == min(mean_absolute_error, na.rm = TRUE)) %>%
    ungroup()

  list(
    hindcast = hindcast,
    observed_data = observed_data,
    scores = scores,
    best_fitness = best_fitness
  )
}
