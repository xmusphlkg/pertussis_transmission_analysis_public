## Figure 1 data preparation ---------------------------------------------------

figure_1_selected_regions <- function() {
  c(
    "European Region",
    "Western Pacific Region",
    "Eastern Mediterranean Region",
    "South-East Asia Region",
    "Region of the Americas",
    "African Region"
  )
}

figure_1_region_display_labels <- function() {
  c(
    "Global" = "Global",
    "Western Pacific Region" = "Western\nPacific",
    "European Region" = "Europe",
    "Eastern Mediterranean Region" = "Eastern\nMediterranean",
    "Region of the Americas" = "Americas",
    "African Region" = "Africa",
    "South-East Asia Region" = "South-East\nAsia"
  )
}

figure_1_country_who_region <- function() {
  country_who_regions
}

figure_1b_interval_contract <- function() {
  settings_path <- model_path("config", "model_settings.yaml")
  if (!file.exists(settings_path)) {
    stop("Figure 1b intervals require current model settings.", call. = FALSE)
  }
  settings <- yaml::read_yaml(settings_path)
  contract <- settings$runtime$bayesian_uncertainty$figure1b_current_practice_conditional_parametric_bootstrap_confidence_interval
  bootstrap_replicates <- as.integer(contract$replicates_per_country)
  minimum_successful_replicates <- as.integer(contract$minimum_successful_replicates)
  if (length(bootstrap_replicates) != 1L ||
      length(minimum_successful_replicates) != 1L ||
      is.na(bootstrap_replicates) ||
      is.na(minimum_successful_replicates) ||
      minimum_successful_replicates <= 0L ||
      bootstrap_replicates < minimum_successful_replicates) {
    stop("Figure 1b interval replicate settings are invalid.", call. = FALSE)
  }
  list(
    bootstrap_replicates = bootstrap_replicates,
    minimum_successful_replicates = minimum_successful_replicates
  )
}

validate_figure_1b_interval_metadata <- function(interval_metadata) {
  if (!identical(
        interval_metadata$bootstrap_data_generation,
        "conditional_fitted_AR1_path_plus_NB2_measurement"
      ) ||
      !isTRUE(interval_metadata$conditional_on_fitted_latent_process_path) ||
      !identical(interval_metadata$latent_process_path_regenerated, FALSE) ||
      !identical(
        interval_metadata$bootstrap_refit,
        "country_state_space_MAP_full_refit_per_replicate"
      )) {
    stop(
      "Figure 1b bootstrap metadata do not identify the conditional fitted-path route.",
      call. = FALSE
    )
  }
  invisible(interval_metadata)
}

load_figure_1b_interval_metadata <- function() {
  interval_path <- model_path(
    "outputs", "metadata",
    "figure1b_current_practice_conditional_parametric_bootstrap_run_metadata.json"
  )
  if (!file.exists(interval_path)) {
    stop("Figure 1b requires bootstrap run metadata.", call. = FALSE)
  }
  interval_metadata <- jsonlite::read_json(interval_path, simplifyVector = TRUE)
  validate_figure_1b_interval_metadata(interval_metadata)
}

prepare_figure_1b_endpoint_intervals <- function(
  interval_audit,
  publication_countries,
  bootstrap_replicates,
  minimum_successful_replicates
) {
  require_columns(
    interval_audit,
    c(
      "country", "outcome", "rate_q025", "rate_q975",
      "bootstrap_replicates", "interval_type", "interval_basis",
      "confidence_interval_method"
    ),
    "figure1b_current_practice_conditional_confidence_intervals.csv"
  )
  outcome_order <- c("Reports", "Symptomatic", "Infections")
  intervals <- interval_audit %>%
    transmute(
      country = stringr::str_replace_all(as.character(country), " ", "_"),
      outcome = as.character(outcome),
      interval_lower_per_100k = as.numeric(rate_q025),
      interval_upper_per_100k = as.numeric(rate_q975),
      bootstrap_replicates = as.integer(bootstrap_replicates),
      interval_type = as.character(interval_type),
      interval_basis = as.character(interval_basis),
      confidence_interval_method = as.character(confidence_interval_method)
    )

  expected_countries <- unique(stringr::str_replace_all(as.character(publication_countries), " ", "_"))
  if (!setequal(unique(intervals$country), expected_countries) ||
      !setequal(unique(intervals$outcome), outcome_order) ||
      nrow(intervals) != length(expected_countries) * length(outcome_order) ||
      anyDuplicated(intervals[c("country", "outcome")]) > 0L ||
      any(!is.finite(intervals$interval_lower_per_100k)) ||
      any(!is.finite(intervals$interval_upper_per_100k)) ||
      any(intervals$interval_lower_per_100k < 0) ||
      any(intervals$interval_lower_per_100k > intervals$interval_upper_per_100k) ||
      any(intervals$bootstrap_replicates < minimum_successful_replicates) ||
      any(intervals$bootstrap_replicates > bootstrap_replicates)) {
    stop("Figure 1b requires complete current-practice confidence intervals for all endpoints.", call. = FALSE)
  }
  if (!identical(unique(intervals$interval_type),
                 "95% parametric-bootstrap confidence interval") ||
      !identical(unique(intervals$confidence_interval_method),
                 "percentile_parametric_bootstrap") ||
      any(!stringr::str_detect(stringr::str_to_lower(intervals$interval_basis), "parametric bootstrap")) ||
      any(!stringr::str_detect(stringr::str_to_lower(intervals$interval_basis), "conditional")) ||
      any(!stringr::str_detect(stringr::str_to_lower(intervals$interval_basis), "fitted annual latent transmission path")) ||
      any(!stringr::str_detect(stringr::str_to_lower(intervals$interval_basis), "fully refitted")) ||
      any(stringr::str_detect(
        stringr::str_to_lower(intervals$interval_basis),
        "marginal|regenerates a complete annual latent transmission path|posterior|credible|prediction interval|predictive interval"
      ))) {
    stop("Figure 1b interval provenance is not the audited conditional full-refit parametric bootstrap.", call. = FALSE)
  }

  intervals %>%
    arrange(match(country, expected_countries), match(outcome, outcome_order))
}

load_figure_1_inputs <- function() {
  interval_contract <- figure_1b_interval_contract()
  load_figure_1b_interval_metadata()
  list(
    regional_incidence = readr::read_csv(
      model_path("data", "processed", "who_pertussis_region_incidence.csv"),
      show_col_types = FALSE
    ),
    pediatric_burden = read_table("lancet_baseline_pediatric_burden.csv"),
    intervention_summary = readr::read_csv(
      model_path("outputs", "summaries", "intervention_scenarios_summary.csv"),
      show_col_types = FALSE
    ),
    primary_interval_audit = read_summary("figure1b_current_practice_conditional_confidence_intervals.csv"),
    bootstrap_replicates = interval_contract$bootstrap_replicates,
    minimum_successful_replicates = interval_contract$minimum_successful_replicates
  )
}

prepare_figure_1_data <- function(inputs = load_figure_1_inputs()) {
  selected_regions <- figure_1_selected_regions()
  region_labels <- figure_1_region_display_labels()

  regional_incidence <- inputs$regional_incidence %>%
    required_columns(
      c("region", "year", "reported_incidence_per_million"),
      "WHO regional incidence input"
    ) %>%
    mutate(reported_incidence_per_100k = reported_incidence_per_million / 10)

  expected_regions <- c("Global", selected_regions)
  missing_regions <- setdiff(expected_regions, unique(regional_incidence$region))
  missing_labels <- setdiff(expected_regions, names(region_labels))
  missing_colours <- setdiff(expected_regions, names(region_colours))
  if (length(missing_regions) > 0L ||
      length(missing_labels) > 0L ||
      length(missing_colours) > 0L) {
    stop(
      "Figure 1a requires Global plus all six WHO regions with complete labels and colours.",
      call. = FALSE
    )
  }

  regional_focus <- regional_incidence %>%
    filter(region %in% expected_regions) %>%
    mutate(region = factor(region, levels = expected_regions))

  regional_focus_latest <- regional_focus %>%
    group_by(region) %>%
    filter(positive_rate(reported_incidence_per_100k)) %>%
    slice_max(year, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(region_label = region_labels[as.character(region)])

  pediatric_burden <- inputs$pediatric_burden %>%
    required_columns(
      c(
        "country", "primary_cases_per_100k", "infant_cases_per_100k",
        "infant_hospitalizations_per_100k", "infant_deaths_per_100k",
        "child_1_9_cases_per_100k", "adolescent_cases_per_100k",
        "child_adolescent_reported_cases_per_100k", "child_adolescent_infections_per_100k",
        "age_case_data_availability", "lancet_endpoint_validation_tier"
      ),
      "Lancet baseline pediatric burden table"
    ) %>%
    mutate(country = stringr::str_replace_all(country, " ", "_"))

  available_country_labels <- unique(format_country(pediatric_burden$country))
  country_order <- main_figure_country_order(available_country_labels)
  if (length(country_order) != length(main_figure_country_label_levels) ||
      !setequal(country_order, available_country_labels)) {
    stop("Figure 1 countries do not match the fixed main-figure order.", call. = FALSE)
  }

  country_who_region <- figure_1_country_who_region()

  decision_map <- pediatric_burden %>%
    mutate(
      country_label_text = format_country(country),
      country_code = unname(country_codes[country]),
      who_region = unname(country_who_region[country]),
      who_region = if_else(is.na(who_region), "Other", who_region),
      primary_cases_per_100k = as.numeric(primary_cases_per_100k),
      infant_cases_per_100k = as.numeric(infant_cases_per_100k),
      infant_hospitalizations_per_100k = as.numeric(infant_hospitalizations_per_100k),
      infant_deaths_per_100k = as.numeric(infant_deaths_per_100k)
    ) %>%
    filter(
      positive_rate(primary_cases_per_100k),
      positive_rate(infant_hospitalizations_per_100k)
    ) %>%
    distinct(country, .keep_all = TRUE)

  decision_medians <- decision_map %>%
    summarise(
      median_primary_cases_per_100k = median(primary_cases_per_100k, na.rm = TRUE),
      median_infant_hospitalizations_per_100k = median(infant_hospitalizations_per_100k, na.rm = TRUE),
      median_infant_deaths_per_100k = median(infant_deaths_per_100k, na.rm = TRUE)
    )

  baseline_composition <- inputs$intervention_summary %>%
    required_columns(
      c(
        "country", "scenario", "total_infant_cases", "total_child_1_9_cases",
        "total_adolescent_cases", "total_child_adolescent_cases",
        "child_adolescent_population", "analysis_years"
      ),
      "Intervention scenario summary"
    ) %>%
    filter(scenario == "current") %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_label = factor(country_label_text, levels = rev(country_order)),
      country_code = unname(country_codes[country]),
      total_infant_cases = as.numeric(total_infant_cases),
      total_child_1_9_cases = as.numeric(total_child_1_9_cases),
      total_adolescent_cases = as.numeric(total_adolescent_cases),
      total_child_adolescent_cases = as.numeric(total_child_adolescent_cases),
      child_adolescent_population = as.numeric(child_adolescent_population),
      analysis_years = as.numeric(analysis_years)
    ) %>%
    filter(
      !is.na(country_label),
      positive_rate(total_child_adolescent_cases),
      positive_rate(child_adolescent_population),
      positive_rate(analysis_years)
    ) %>%
    distinct(country, .keep_all = TRUE) %>%
    select(
      country, country_code, country_label, child_adolescent_population, analysis_years,
      total_infant_cases, total_child_1_9_cases, total_adolescent_cases,
      total_child_adolescent_cases
    ) %>%
    pivot_longer(
      c(total_infant_cases, total_child_1_9_cases, total_adolescent_cases),
      names_to = "age_group",
      values_to = "symptomatic_cases"
    ) %>%
    mutate(
      age_group = factor(
        age_group,
        levels = c("total_infant_cases", "total_child_1_9_cases", "total_adolescent_cases"),
        labels = c("Infant", "Children", "Adolescent")
      ),
      burden_share = symptomatic_cases / total_child_adolescent_cases,
      symptomatic_cases_per_100k_under18 = symptomatic_cases / analysis_years / child_adolescent_population * 1e5
    )

  baseline_endpoints <- inputs$intervention_summary %>%
    required_columns(
      c(
        "country", "scenario", "child_adolescent_population", "analysis_years",
        "total_child_adolescent_cases", "total_child_adolescent_reported_cases",
        "total_child_adolescent_infections",
        "annualized_child_adolescent_cases_per_100k",
        "annualized_child_adolescent_reported_cases_per_100k",
        "annualized_child_adolescent_infections_per_100k"
      ),
      "Intervention scenario summary"
    ) %>%
    filter(scenario == "current") %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_label = factor(country_label_text, levels = rev(country_order)),
      country_code = unname(country_codes[country]),
      child_adolescent_population = as.numeric(child_adolescent_population),
      analysis_years = as.numeric(analysis_years)
    ) %>%
    filter(!is.na(country_label)) %>%
    distinct(country, .keep_all = TRUE)
  if (nrow(baseline_endpoints) != length(country_order) ||
      anyDuplicated(baseline_endpoints$country) > 0L ||
      !setequal(format_country(baseline_endpoints$country), country_order)) {
    stop("Figure 1b requires one current-practice row per publication profile.", call. = FALSE)
  }

  baseline_endpoint_long <- baseline_endpoints %>%
    select(
      country,
      country_code,
      country_label_text,
      country_label,
      child_adolescent_population,
      analysis_years,
      total_child_adolescent_cases,
      total_child_adolescent_reported_cases,
      total_child_adolescent_infections,
      annualized_child_adolescent_cases_per_100k,
      annualized_child_adolescent_reported_cases_per_100k,
      annualized_child_adolescent_infections_per_100k
    ) %>%
    pivot_longer(
      cols = c(
        annualized_child_adolescent_cases_per_100k,
        annualized_child_adolescent_reported_cases_per_100k,
        annualized_child_adolescent_infections_per_100k
      ),
      names_to = "outcome",
      values_to = "rate_per_100k"
    ) %>%
    mutate(
      outcome = factor(
        outcome,
        levels = c(
          "annualized_child_adolescent_reported_cases_per_100k",
          "annualized_child_adolescent_cases_per_100k",
          "annualized_child_adolescent_infections_per_100k"
        ),
        labels = c("Reports", "Symptomatic", "Infections")
      ),
      outcome_offset = recode(
        as.character(outcome),
        "Reports" = -0.16,
        "Symptomatic" = 0,
        "Infections" = 0.16,
        .default = 0
      ),
      country_y_base = as.numeric(factor(country_label_text, levels = rev(country_order))),
      country_y = country_y_base + outcome_offset
    ) %>%
    filter(positive_rate(rate_per_100k))
  if (nrow(baseline_endpoint_long) != 3L * length(country_order) ||
      anyDuplicated(baseline_endpoint_long[c("country", "outcome")]) > 0L ||
      any(!is.finite(baseline_endpoint_long$rate_per_100k))) {
    stop("Figure 1b requires three finite current-practice point indices per profile.", call. = FALSE)
  }

  endpoint_intervals <- prepare_figure_1b_endpoint_intervals(
    inputs$primary_interval_audit,
    publication_countries = baseline_endpoints$country,
    bootstrap_replicates = inputs$bootstrap_replicates,
    minimum_successful_replicates = inputs$minimum_successful_replicates
  ) %>%
    left_join(
      baseline_endpoint_long %>%
        mutate(outcome = as.character(outcome)) %>%
        select(country, outcome, country_label_text, country_y, rate_per_100k),
      by = c("country", "outcome")
    )
  if (nrow(endpoint_intervals) != 3L * length(country_order) ||
      anyDuplicated(endpoint_intervals[c("country", "outcome")]) > 0L ||
      any(!is.finite(endpoint_intervals$country_y)) ||
      any(!is.finite(endpoint_intervals$rate_per_100k)) ||
      any(endpoint_intervals$rate_per_100k < endpoint_intervals$interval_lower_per_100k) ||
      any(endpoint_intervals$rate_per_100k > endpoint_intervals$interval_upper_per_100k)) {
    stop("Figure 1b deterministic endpoint indices are not covered by valid matched intervals.", call. = FALSE)
  }

  baseline_endpoint_long <- baseline_endpoint_long %>%
    left_join(
      endpoint_intervals %>%
        select(
          country,
          outcome,
          interval_lower_per_100k,
          interval_upper_per_100k,
          bootstrap_replicates,
          interval_type,
          interval_basis,
          confidence_interval_method
        ),
      by = c("country", "outcome")
    ) %>%
    mutate(
      interval_applies = TRUE,
      across(
        c(interval_lower_per_100k, interval_upper_per_100k, bootstrap_replicates),
        ~ if_else(interval_applies, ., NA_real_)
      ),
      across(
        c(interval_type, interval_basis, confidence_interval_method),
        ~ if_else(interval_applies, ., NA_character_)
      )
    )

  endpoint_spans <- baseline_endpoint_long %>%
    group_by(country, country_label_text, country_y_base) %>%
    summarise(
      min_rate_per_100k = min(rate_per_100k, na.rm = TRUE),
      max_rate_per_100k = max(rate_per_100k, na.rm = TRUE),
      .groups = "drop"
    )

  list(
    selected_regions = selected_regions,
    region_labels = region_labels,
    regional_focus = regional_focus,
    regional_focus_latest = regional_focus_latest,
    pediatric_burden = pediatric_burden,
    country_order = country_order,
    decision_map = decision_map,
    decision_medians = decision_medians,
    baseline_composition = baseline_composition,
    baseline_endpoint_long = baseline_endpoint_long,
    endpoint_spans = endpoint_spans,
    endpoint_intervals = endpoint_intervals
  )
}
