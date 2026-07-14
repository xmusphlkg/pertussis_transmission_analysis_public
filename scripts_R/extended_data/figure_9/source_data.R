## Extended Data Figure 9 source data -----------------------------------------

extended_data_figure_9_source_filenames <- c(
  primary_under18_programme_rank_sensitivity = "extended_data_figure_9e_primary_under18_programme_rank_sensitivity.csv",
  age_pattern_programme_robustness = "extended_data_figure_9g_age_pattern_programme_robustness.csv"
)

make_extended_data_figure_9_source_data <- function(data) {
  list(
    primary_under18_programme_rank_sensitivity = data$psa %>%
      transmute(
        country,
        strategy,
        strategy_label = as.character(strategy_label),
        frequency_rank_1,
        frequency_top_2,
        frequency_top_3,
        frequency_within_10_percent_of_best,
        mean_rank,
        median_rank,
        median_under18_cases_per_100k,
        q025_under18_cases_per_100k,
        q975_under18_cases_per_100k,
        median_relative_reduction_vs_current,
        n_psa_samples,
        n_rank_observations,
        interpretation
      ),
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
