## Figure 4 source data --------------------------------------------------------

figure_4_source_filenames <- c(
  mechanism_contrasts = "figure4a_resistance_mechanism_contrasts.csv",
  management_tradeoff = "figure4b_resistance_management_tradeoff.csv",
  preference_weight_summary = "figure4c_resistance_preference_weight_summary.csv",
  preference_country_thresholds = "figure4c_resistance_preference_country_thresholds.csv",
  implementation_sensitivity = "figure4_resistance_implementation_sensitivity_appendix.csv",
  future_vaccine_residual_burden = "figure4c_future_vaccine_residual_burden.csv",
  future_veinf_thresholds = "figure4d_future_veinf_thresholds.csv",
  future_veinf_attainment = "figure4d_future_veinf_attainment.csv",
  future_veinf_seeded_attainment_summary = "figure4d_future_veinf_seeded_attainment_summary.csv"
)

make_figure_4_source_data <- function(data) {
  list(
    mechanism_contrasts = data$mechanism_delta %>%
      left_join(
        data$mechanism_delta_summary %>%
          select(
            scenario,
            mechanism_label,
            metric,
            median_reduction_vs_baseline,
            q25_reduction_vs_baseline,
            q75_reduction_vs_baseline
          ),
        by = c("scenario", "mechanism_label", "metric")
      ) %>%
      transmute(
        country,
        scenario,
        mechanism_label = as.character(mechanism_label),
        mechanism_axis_label = unname(data$mechanism_axis_labels[as.character(mechanism_label)]),
        metric = as.character(metric),
        mechanism_y,
        mechanism_y_metric,
        reduction_vs_baseline,
        median_reduction_vs_baseline,
        q25_reduction_vs_baseline,
        q75_reduction_vs_baseline,
        resistant_infections_per_100k,
        baseline_resistant_infections_per_100k,
        child_adolescent_cases_per_100k,
        baseline_child_adolescent_cases_per_100k,
        end_resistant_fraction = resistant_fraction_end,
        resistant_importation,
        treatment_differential,
        pep_differential,
        fitness_R,
        interpretation = mechanism_interpretation
      ),
    management_tradeoff = data$resistance_management %>%
      transmute(
        country,
        country_code,
        current_resistant_infections_per_100k,
        remaining_resistant_infections_per_100k,
        absolute_resistant_infections_averted_per_100k =
          current_resistant_infections_per_100k - remaining_resistant_infections_per_100k,
        remaining_child_adolescent_cases_per_100k = primary_cases_per_100k,
        primary_case_reduction,
        relative_reduction_resistant_infections,
        resistance_reduction_estimable
      ),
    preference_weight_summary = data$preference_summary %>%
      transmute(
        resistance_weight_lambda,
        strategy,
        strategy_label = as.character(strategy_label),
        countries_preferred,
        median_preference_score,
        median_infant_case_reduction,
        median_resistant_infection_reduction
      ),
    preference_country_thresholds = data$preference_thresholds,
    implementation_sensitivity = data$implementation_plot %>%
      mutate(
        pep_assumption = as.character(pep_assumption),
        reach_status = as.character(reach_status),
        implementation_group = as.character(implementation_group),
        metric = as.character(metric)
      ),
    future_vaccine_residual_burden = data$vaccine_residual %>%
      mutate(scenario_label = as.character(scenario_label)) %>%
      left_join(data$vaccine_burden_summary, by = "scenario_label") %>%
      transmute(
        country,
        country_code,
        country_label_text,
        scenario,
        scenario_label,
        outcome_type = as.character(outcome_type),
        primary_cases_per_100k,
        min_primary_cases_per_100k,
        max_primary_cases_per_100k,
        median,
        q25,
        q75,
        q025,
        q975
      ),
    future_veinf_thresholds = data$veinf_threshold_source,
    future_veinf_attainment = data$veinf_attainment_source,
    future_veinf_seeded_attainment_summary = data$veinf_attainment_plot_summary %>%
      transmute(
        comparator,
        comparator_label = as.character(comparator_label),
        starting_resistant_fraction,
        seed_label = as.character(seed_label),
        infectiousness_reduction,
        profiles_matching_comparator,
        countries_evaluated
      )
  )
}

write_figure_4_source_data <- function(source_data) {
  write_source_data_list(source_data, figure_4_source_filenames)
}
