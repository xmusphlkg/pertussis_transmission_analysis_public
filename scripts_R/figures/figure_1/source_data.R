## Figure 1 source data --------------------------------------------------------

figure_1_source_filenames <- c(
  burden_layers = "figure1b_burden_layers.csv",
  decision_map = "figure1c_current_practice_decision_map.csv",
  age_composition = "figure1d_baseline_age_composition.csv"
)

make_figure_1_source_data <- function(data) {
  list(
    burden_layers = data$baseline_endpoint_long %>%
      transmute(
        country,
        country_code,
        outcome = as.character(outcome),
        child_adolescent_population,
        analysis_years,
        total_child_adolescent_cases,
        total_child_adolescent_reported_cases,
        total_child_adolescent_infections,
        rate_per_100k
      ),
    decision_map = data$decision_map %>%
      transmute(
        country,
        country_code,
        who_region,
        primary_cases_per_100k,
        infant_cases_per_100k,
        infant_hospitalizations_per_100k,
        infant_deaths_per_100k,
        median_primary_cases_per_100k = data$decision_medians$median_primary_cases_per_100k[[1]],
        median_infant_hospitalizations_per_100k = data$decision_medians$median_infant_hospitalizations_per_100k[[1]],
        median_infant_deaths_per_100k = data$decision_medians$median_infant_deaths_per_100k[[1]]
      ),
    age_composition = data$baseline_composition %>%
      transmute(
        country,
        country_code,
        age_group = as.character(age_group),
        symptomatic_cases,
        total_child_adolescent_cases,
        child_adolescent_population,
        analysis_years,
        symptomatic_cases_per_100k_under18,
        burden_share
      )
  )
}

write_figure_1_source_data <- function(source_data) {
  write_source_data_list(source_data, figure_1_source_filenames)
}
