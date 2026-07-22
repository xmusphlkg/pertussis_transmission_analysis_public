## Figure 3 source data --------------------------------------------------------

figure_3_source_filenames <- c(
  endpoint_effect_matrix = "figure3a_endpoint_effect_matrix.csv",
  infant_gap = "figure3b_infant_to_child_adolescent_gap.csv",
  age_contribution = "figure3c_age_contribution_averted_cases.csv",
  adolescent_booster = "figure3d_adolescent_booster_profile_effects.csv"
)

make_figure_3_source_data <- function(data) {
  list(
    endpoint_effect_matrix = data$endpoint_country %>%
      left_join(
        data$endpoint_effect_matrix %>%
          select(
            strategy,
            endpoint,
            median_relative_case_reduction,
            q25_relative_case_reduction,
            q75_relative_case_reduction,
            min_relative_case_reduction,
            max_relative_case_reduction
          ),
        by = c("strategy", "endpoint")
      ) %>%
      transmute(
        country,
        strategy,
        strategy_label,
        outcome = as.character(endpoint),
        relative_case_reduction,
        median_relative_case_reduction,
        q25_relative_case_reduction,
        q75_relative_case_reduction,
        min_relative_case_reduction,
        max_relative_case_reduction
      ),
    infant_gap = data$endpoint_gap %>%
      left_join(
        data$endpoint_gap_summary %>%
          select(strategy, median_gap_pp, q25_gap_pp, q75_gap_pp, min_gap_pp, max_gap_pp),
        by = "strategy"
      ) %>%
      transmute(
        country,
        strategy,
        strategy_label,
        infant_case_reduction,
        overall_case_reduction = all_under18_case_reduction,
        infant_minus_overall_gap_pp = infant_minus_child_adolescent_gap_pp,
        median_gap_pp,
        q25_gap_pp,
        q75_gap_pp,
        min_gap_pp,
        max_gap_pp
      ),
    age_contribution = data$age_contribution_long %>%
      left_join(
        data$age_contribution_summary %>%
          select(
            strategy,
            age_contribution,
            median_cases_averted_per_100k_under18,
            q25_cases_averted_per_100k_under18,
            q75_cases_averted_per_100k_under18
          ),
        by = c("strategy", "age_contribution")
      ) %>%
      transmute(
        country,
        strategy,
        strategy_label,
        age_group = as.character(age_contribution),
        cases_averted_per_100k_aged_under_18 = cases_averted_per_100k_under18,
        overall_cases_averted_per_100k_aged_under_18 = primary_averted_per_100k_under18,
        age_group_sum_averted_per_100k_aged_under_18 = component_sum_averted_per_100k_under18,
        component_sum_difference_per_100k,
        median_cases_averted_per_100k_aged_under_18 = median_cases_averted_per_100k_under18,
        q25_cases_averted_per_100k_aged_under_18 = q25_cases_averted_per_100k_under18,
        q75_cases_averted_per_100k_aged_under_18 = q75_cases_averted_per_100k_under18
      ),
    adolescent_booster = data$booster_effect_profile %>%
      transmute(
        country,
        country_code,
        adolescent_effect_rank,
        all_under18_symptomatic_case_reduction,
        adolescent_case_reduction,
        all_under18_cases_per_100k,
        adolescent_cases_per_100k,
        current_all_under18_cases_per_100k,
        current_adolescent_cases_per_100k
      )
  )
}

write_figure_3_source_data <- function(source_data) {
  write_source_data_list(source_data, figure_3_source_filenames)
}
