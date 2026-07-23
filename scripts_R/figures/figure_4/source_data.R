## Figure 4 source data --------------------------------------------------------

figure_4_source_filenames <- c(
  management_pairwise = "figure4a_resistance_guided_vs_timeliness.csv",
  vaccine_residual = "figure4c_vaccine_setting_residual_index.csv"
)

figure_4_legacy_source_filenames <- c(
  vaccine_residual = "figure4b_vaccine_setting_residual_index.csv"
)

make_figure_4_source_data <- function(data) {
  list(
    management_pairwise = data$management_pairwise %>%
      left_join(data$management_summary, by = "endpoint") %>%
      transmute(
        country,
        country_name,
        outcome = as.character(endpoint),
        routine_timeliness_raw_burden = routine_burden,
        resistance_guided_raw_burden = guided_burden,
        guided_residual_index_vs_routine = residual_index,
        guided_reduction_vs_routine = reduction_vs_routine,
        direction,
        profiles_guided_lower,
        profiles_routine_lower,
        median_residual_index,
        q25_residual_index,
        q75_residual_index,
        min_residual_index,
        max_residual_index,
        burden_unit = "annualised per 100,000 endpoint-specific population",
        analysis_horizon = "2027-01-01 to 2050-12-31",
        contrast_definition
      ),
    vaccine_residual = data$vaccine_outcomes %>%
      left_join(
        data$vaccine_summary_matrix %>%
          select(
            scenario, endpoint, profiles_evaluated, profiles_above_current,
            median_residual_index, q25_residual_index, q75_residual_index,
            min_residual_index, max_residual_index
          ),
        by = c("scenario", "endpoint")
      ) %>%
      left_join(data$vaccine_parameters, by = "scenario") %>%
      transmute(
        country,
        country_name = format_country(country),
        vaccine_setting = scenario,
        vaccine_setting_label = scenario_label_text,
        VE_sus,
        VE_sym,
        VE_inf,
        VE_dur,
        history_vaccine_scenario,
        vaccine_transition_design,
        vaccine_transition_interpretation,
        outcome = as.character(endpoint),
        current_ap_like_raw_burden = current_burden,
        scenario_raw_burden = scenario_burden,
        residual_index_vs_current = residual_index,
        reduction_vs_current,
        above_current,
        profiles_evaluated,
        profiles_above_current,
        median_residual_index,
        q25_residual_index,
        q75_residual_index,
        min_residual_index,
        max_residual_index,
        burden_unit = "cumulative event count",
        analysis_horizon = "2027-01-01 to 2050-12-31",
        contrast_definition
      )
  )
}

write_figure_4_source_data <- function(source_data) {
  write_source_data_list(source_data, figure_4_source_filenames)
  write_source_data_list(
    source_data[names(figure_4_legacy_source_filenames)],
    figure_4_legacy_source_filenames
  )
}
