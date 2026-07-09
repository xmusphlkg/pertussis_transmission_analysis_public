## Figure 2 source data --------------------------------------------------------

figure_2_source_filenames <- c(
  programme_effects = "figure2a_programme_effects.csv",
  country_strategy_matrix = "figure2b_programme_country_strategy_matrix.csv",
  selected_burden = "figure2c_programme_selected_burden.csv",
  rank_distribution = "figure2c_programme_rank_distribution.csv",
  runner_up_excess = "figure2d_programme_runner_up_excess.csv"
)

make_figure_2_source_data <- function(data) {
  list(
    programme_effects = data$program_frontier %>%
      transmute(
        country,
        strategy,
        strategy_label,
        primary_case_reduction,
        primary_cases_per_100k,
        implementation_intensity,
        primary_case_rank_within_programme_only = decision_rank,
        not_clearly_surpassed = non_dominated_lancet_outcome,
        lowest_burden_in_programme_only = decision_rank == 1
      ),
    country_strategy_matrix = data$program_heatmap %>%
      transmute(
        country,
        strategy,
        strategy_label,
        primary_case_reduction,
        primary_cases_per_100k,
        current_primary_cases_per_100k,
        implementation_intensity,
        primary_case_rank_within_programme_only = decision_rank,
        margin_to_best_cases_per_100k,
        margin_to_best_percentage_points,
        near_best_5_cases,
        lowest_burden_in_programme_only = preferred_in_program_only,
        not_clearly_surpassed = non_dominated_lancet_outcome,
        primary_reduction_interval_q025,
        primary_reduction_interval_q975,
        current_rate_interval_q025,
        current_rate_interval_q975,
        intervention_rate_interval_q025,
        intervention_rate_interval_q975,
        interval_basis
      ),
    selected_burden = data$selected_program %>%
      transmute(
        country,
        current_cases_per_100k,
        lowest_burden_strategy = winning_strategy,
        lowest_burden_strategy_label = winning_strategy_label,
        lowest_burden_reduction = winning_reduction,
        lowest_burden_cases_per_100k = winning_cases_per_100k,
        cases_averted_per_100k_with_lowest_burden_option = selected_cases_averted_per_100k,
        runner_up_strategy,
        runner_up_strategy_label,
        runner_up_reduction,
        runner_up_cases_per_100k,
        runner_up_excess_reduction = winner_margin_reduction,
        runner_up_excess_percentage_points = 100 * winner_margin_reduction,
        runner_up_excess_cases_per_100k = winner_margin_cases_per_100k
      ),
    rank_distribution = data$rank_distribution %>%
      transmute(
        strategy,
        strategy_label,
        within_profile_rank = decision_rank,
        rank_label = as.character(rank_label),
        profile_count,
        total_profiles = rank_total_profiles
      ),
    runner_up_excess = data$selected_program %>%
      transmute(
        country,
        lowest_burden_strategy = winning_strategy,
        lowest_burden_strategy_label = winning_strategy_label,
        runner_up_strategy,
        runner_up_strategy_label,
        lowest_burden_cases_per_100k = winning_cases_per_100k,
        runner_up_cases_per_100k,
        runner_up_excess_cases_per_100k = winner_margin_cases_per_100k,
        runner_up_excess_reduction = winner_margin_reduction,
        runner_up_excess_percentage_points = 100 * winner_margin_reduction
      )
  )
}

write_figure_2_source_data <- function(source_data) {
  write_source_data_list(source_data, figure_2_source_filenames)
}
