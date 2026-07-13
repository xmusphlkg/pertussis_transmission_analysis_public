## Extended Data Figure 2 data preparation ------------------------------------

extended_data_figure_2_country_who_region <- function() {
  c(
    Australia = "Western Pacific Region",
    China = "Western Pacific Region",
    Japan = "Western Pacific Region",
    New_Zealand = "Western Pacific Region",
    Sweden = "European Region",
    United_Kingdom = "European Region",
    United_States = "Region of the Americas",
    Brazil = "Region of the Americas",
    Thailand = "South-East Asia Region",
    South_Africa = "African Region"
  )
}

extended_data_figure_2_input_paths <- function() {
  list(
    observed = model_path("data", "processed", "pertussis_incidence_timeseries.csv"),
    baseline = model_path("outputs", "summaries", "country_scenarios_summary"),
    calibration = model_path("outputs", "tables", "calibration_all_countries.csv"),
    baseline_pediatric_burden = model_path("outputs", "tables", "lancet_baseline_pediatric_burden.csv"),
    age_pattern_current = model_path("outputs", "tables", "lancet_age_pattern_fit_current.csv"),
    age_pattern_fallback = model_path("outputs", "tables", "age_pattern_country_weights.csv"),
    reporting_summary = model_path("outputs", "summaries", "reporting_scenarios_summary")
  )
}

load_extended_data_figure_2_inputs <- function(paths = extended_data_figure_2_input_paths()) {
  if (!file.exists(paths$calibration)) {
    stop(
      "eFigure 2 publication rendering requires outputs/tables/calibration_all_countries.csv from `make calibrate`.",
      call. = FALSE
    )
  }

  age_pattern_path <- paths$age_pattern_current
  if (!file.exists(age_pattern_path)) {
    age_pattern_path <- paths$age_pattern_fallback
  }
  if (!file.exists(age_pattern_path)) {
    stop(
      "eFigure 2 panel F requires external age-pattern checks from outputs/tables/lancet_age_pattern_fit_current.csv.",
      call. = FALSE
    )
  }

  list(
    observed = readr::read_csv(paths$observed, show_col_types = FALSE),
    baseline = read_model_table(paths$baseline),
    calibration = readr::read_csv(paths$calibration, show_col_types = FALSE),
    baseline_pediatric_burden = readr::read_csv(paths$baseline_pediatric_burden, show_col_types = FALSE),
    age_pattern = readr::read_csv(age_pattern_path, show_col_types = FALSE),
    age_pattern_path = age_pattern_path,
    reporting_summary = read_model_table(paths$reporting_summary)
  )
}

prepare_extended_data_figure_2_data <- function(inputs = load_extended_data_figure_2_inputs()) {
  country_who_region <- extended_data_figure_2_country_who_region()

  baseline <- inputs$baseline %>%
    add_country_label()

  observed <- inputs$observed
  observed_annual <- observed %>%
    mutate(country = config_key) %>%
    group_by(country, Year) %>%
    summarise(observed_cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
    left_join(baseline %>% select(country, total_population), by = "country") %>%
    add_country_label() %>%
    mutate(
      observed_reported_incidence = observed_cases / total_population * 1e5,
      country_code = factor(country_codes[country], levels = country_codes[country_levels])
    )

  calibration <- inputs$calibration
  if ("calibration_accepted" %in% names(calibration) && !"calibration_success" %in% names(calibration)) {
    calibration$calibration_success <- tolower(as.character(calibration$calibration_accepted)) %in% c("true", "1", "yes", "accepted")
  }

  require_columns(
    calibration,
    c(
      "country",
      "fitted_temporal_mean_reported_cases",
      "fitted_temporal_range_low",
      "fitted_temporal_range_high",
      "fitted_temporal_range_method",
      "calibration_success"
    ),
    "calibration_all_countries.csv"
  )
  calibration_diagnostic <- calibration %>%
    filter(.data$calibration_success == TRUE) %>%
    transmute(
      country = stringr::str_replace_all(country, " ", "_"),
      country_code = factor(country_codes[country], levels = country_codes[country_levels]),
      fitted_temporal_mean_reported_cases = as.numeric(fitted_temporal_mean_reported_cases),
      fitted_temporal_range_low = as.numeric(fitted_temporal_range_low),
      fitted_temporal_range_high = as.numeric(fitted_temporal_range_high),
      fitted_temporal_range_method = as.character(fitted_temporal_range_method),
      calibration_success
    )
  invalid_calibration_diagnostic <- calibration_diagnostic %>%
    filter(
      is.na(.data$country_code) |
        !is.finite(.data$fitted_temporal_mean_reported_cases) |
        !is.finite(.data$fitted_temporal_range_low) |
        !is.finite(.data$fitted_temporal_range_high) |
        .data$fitted_temporal_range_low > .data$fitted_temporal_mean_reported_cases |
        .data$fitted_temporal_mean_reported_cases > .data$fitted_temporal_range_high |
        is.na(.data$fitted_temporal_range_method) |
        .data$fitted_temporal_range_method !=
          "minimum_and_maximum_fitted_reported_cases_across_likelihood_intervals"
    )
  if (nrow(invalid_calibration_diagnostic) > 0) {
    stop(
      "eFigure 2 panel B received invalid fitted temporal ranges for: ",
      paste(invalid_calibration_diagnostic$country, collapse = ", "),
      call. = FALSE
    )
  }
  if (nrow(calibration_diagnostic) == 0) {
    stop(
      "eFigure 2 panel B requires accepted calibration diagnostics with fitted temporal ranges.",
      call. = FALSE
    )
  }

  calibration_country_order <- inputs$baseline_pediatric_burden %>%
    require_columns(c("country", "primary_cases_per_100k"), "Lancet baseline pediatric burden table") %>%
    mutate(country = stringr::str_replace_all(country, " ", "_")) %>%
    arrange(desc(as.numeric(primary_cases_per_100k))) %>%
    mutate(country_label_text = format_country(country)) %>%
    pull(country_label_text)

  require_columns(
    calibration,
    c(
      "country", "observed_mean_annual_reported_incidence_per_100k",
      "annualized_reported_cases_per_100k", "model_to_observed_reported_incidence_ratio",
      "calibration_interval_smape", "calibration_peak_ratio", "calibration_data_overlap_intervals",
      "calibration_log1p_correlation", "calibration_max_interval_smape",
      "calibration_peak_ratio_min", "calibration_peak_ratio_max", "total_population",
      "absolute_fit_relative_tolerance"
    ),
    "calibration_all_countries.csv"
  )

  calibration_fingerprint <- calibration %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_label_factor = factor(country_label_text, levels = rev(calibration_country_order)),
      country_code = unname(country_codes[country]),
      who_region = unname(country_who_region[country]),
      who_region = if_else(is.na(who_region), "Other", who_region),
      observed_reported_incidence_per_100k = as.numeric(observed_mean_annual_reported_incidence_per_100k),
      modelled_reported_incidence_per_100k = as.numeric(annualized_reported_cases_per_100k),
      model_observed_ratio = as.numeric(model_to_observed_reported_incidence_ratio),
      calibration_interval_smape = as.numeric(calibration_interval_smape),
      calibration_peak_ratio = as.numeric(calibration_peak_ratio),
      calibration_log1p_correlation = as.numeric(calibration_log1p_correlation),
      calibration_max_interval_smape = as.numeric(calibration_max_interval_smape),
      calibration_peak_ratio_min = as.numeric(calibration_peak_ratio_min),
      calibration_peak_ratio_max = as.numeric(calibration_peak_ratio_max),
      calibration_data_overlap_intervals = as.numeric(calibration_data_overlap_intervals),
      absolute_fit_relative_tolerance = as.numeric(absolute_fit_relative_tolerance),
      total_population = as.numeric(total_population),
      population_millions = total_population / 1e6,
      mean_ratio_min = 1 - absolute_fit_relative_tolerance,
      mean_ratio_max = 1 + absolute_fit_relative_tolerance
    ) %>%
    filter(
      !is.na(country_label_factor),
      positive_rate(observed_reported_incidence_per_100k),
      positive_rate(modelled_reported_incidence_per_100k),
      positive_rate(model_observed_ratio),
      is.finite(calibration_interval_smape),
      positive_rate(calibration_peak_ratio),
      positive_rate(total_population)
    ) %>%
    distinct(country, .keep_all = TRUE)

  calibration_thresholds <- calibration_fingerprint %>%
    summarise(
      mean_ratio_min = first(na.omit(mean_ratio_min)),
      mean_ratio_max = first(na.omit(mean_ratio_max)),
      max_interval_smape = first(na.omit(calibration_max_interval_smape)),
      peak_ratio_min = first(na.omit(calibration_peak_ratio_min)),
      peak_ratio_max = first(na.omit(calibration_peak_ratio_max))
    )

  if (nrow(calibration_thresholds) == 0 ||
      !is.finite(calibration_thresholds$mean_ratio_min[[1]]) ||
      !is.finite(calibration_thresholds$mean_ratio_max[[1]]) ||
      !is.finite(calibration_thresholds$max_interval_smape[[1]]) ||
      !is.finite(calibration_thresholds$peak_ratio_min[[1]]) ||
      !is.finite(calibration_thresholds$peak_ratio_max[[1]])) {
    stop("Calibration thresholds could not be recovered for eFigure 2.", call. = FALSE)
  }

  age_pattern_checks <- inputs$age_pattern %>%
    require_columns(
      c("country", "country_age_pattern_weight", "passes_weight_threshold", "external_label"),
      basename(inputs$age_pattern_path)
    ) %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_label_factor = factor(country_label_text, levels = rev(calibration_country_order)),
      country_code = unname(country_codes[country]),
      country_age_pattern_weight = as.numeric(country_age_pattern_weight),
      passes_age_pattern_threshold = tolower(as.character(passes_weight_threshold)) %in% c("true", "1", "yes", "pass", "passed"),
      age_pattern_status = factor(
        if_else(passes_age_pattern_threshold, "Pass", "Below threshold"),
        levels = c("Pass", "Below threshold")
      )
    ) %>%
    filter(!is.na(country_label_factor), is.finite(country_age_pattern_weight)) %>%
    distinct(country, .keep_all = TRUE)

  reporting_summary <- inputs$reporting_summary %>%
    add_country_label() %>%
    mutate(
      scenario = factor(scenario, levels = reporting_levels),
      scenario_label = factor(reporting_labels[as.character(scenario)], levels = reporting_labels[reporting_levels])
    )

  require_columns(calibration, c("country", "reporting_multiplier_by_age"), "calibration_all_countries.csv")
  reporting_long <- calibration %>%
    mutate(
      country = str_replace_all(country, " ", "_"),
      country_label = factor(format_country(country), levels = country_label_levels)
    ) %>%
    select(country_label, reporting_multiplier_by_age) %>%
    separate_rows(reporting_multiplier_by_age, sep = ";") %>%
    separate(reporting_multiplier_by_age, into = c("age_group", "reporting_rate"), sep = "=", convert = TRUE) %>%
    mutate(
      age_group = factor(age_group, levels = names(age_labels), labels = age_labels),
      reporting_rate = as.numeric(reporting_rate)
    )

  expected_reporting_cells <- expand_grid(
    country_label = country_label_levels,
    age_group = unname(age_labels)
  )
  observed_reporting_cells <- reporting_long %>%
    transmute(country_label = as.character(country_label), age_group = as.character(age_group)) %>%
    distinct()
  missing_reporting_cells <- expected_reporting_cells %>%
    anti_join(observed_reporting_cells, by = c("country_label", "age_group"))
  if (nrow(missing_reporting_cells) > 0 || any(!is.finite(reporting_long$reporting_rate))) {
    stop(
      "eFigure 2 panel H requires complete fitted age-specific reporting probabilities. Missing cells: ",
      paste(
        paste0(missing_reporting_cells$country_label, "/", missing_reporting_cells$age_group),
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  list(
    observed = observed,
    observed_annual = observed_annual,
    country_who_region = country_who_region,
    calibration = calibration,
    calibration_diagnostic = calibration_diagnostic,
    calibration_country_order = calibration_country_order,
    calibration_fingerprint = calibration_fingerprint,
    calibration_thresholds = calibration_thresholds,
    age_pattern_checks = age_pattern_checks,
    age_pattern_threshold = 0.50,
    reporting_summary = reporting_summary,
    reporting_long = reporting_long
  )
}
