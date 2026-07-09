## Extended Data Figure 9 source data -----------------------------------------

extended_data_figure_9_source_filenames <- c(
  age_pattern_programme_robustness = "extended_data_figure_9g_age_pattern_programme_robustness.csv"
)

make_extended_data_figure_9_source_data <- function(data) {
  list(
    age_pattern_programme_robustness = data$age_pattern_programme %>%
      transmute(
        ordering_basis,
        ordering_basis_label = as.character(ordering_basis_label),
        strategy,
        strategy_label,
        country_count,
        effective_country_weight_sum,
        weighted_median_primary_case_reduction,
        weighted_iqr_low_primary_case_reduction,
        weighted_iqr_high_primary_case_reduction,
        weighted_iqr_primary_case_reduction,
        weighted_median_primary_cases_per_100k,
        strategy_rank_within_basis
      )
  )
}

write_extended_data_figure_9_source_data <- function(source_data) {
  write_source_data_list(source_data, extended_data_figure_9_source_filenames)
}
