## Extended Data Figure 2 source data -----------------------------------------

extended_data_figure_2_source_filenames <- c(
  calibration_fingerprint = "extended_data_figure_2c_calibration_fingerprint.csv",
  age_pattern_checks = "extended_data_figure_2f_age_pattern_checks.csv"
)

make_extended_data_figure_2_source_data <- function(data) {
  list(
    calibration_fingerprint = data$calibration_fingerprint %>%
      transmute(
        country,
        country_code,
        who_region,
        population_millions,
        observed_reported_incidence_per_100k,
        estimated_reported_incidence_per_100k = modelled_reported_incidence_per_100k,
        estimated_observed_ratio = model_observed_ratio,
        calibration_interval_smape,
        calibration_peak_ratio,
        calibration_log1p_correlation,
        absolute_fit_relative_tolerance,
        calibration_max_interval_smape,
        calibration_peak_ratio_min,
        calibration_peak_ratio_max,
        calibration_data_overlap_intervals
      ),
    age_pattern_checks = data$age_pattern_checks %>%
      transmute(
        country,
        country_code,
        external_label,
        country_age_pattern_weight,
        passes_age_pattern_threshold,
        age_pattern_status = as.character(age_pattern_status)
      )
  )
}

write_extended_data_figure_2_source_data <- function(source_data) {
  write_source_data_list(source_data, extended_data_figure_2_source_filenames)
}
