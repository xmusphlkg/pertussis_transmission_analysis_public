## Figure 1 data preparation ---------------------------------------------------

figure_1_selected_regions <- function() {
  c(
    "Western Pacific Region", "European Region",
    "Region of the Americas", "African Region", "South-East Asia Region"
  )
}

figure_1_region_short_labels <- function() {
  c(
    "Global" = "Global",
    "Western Pacific Region" = "WPR",
    "European Region" = "EUR",
    "Region of the Americas" = "AMR",
    "African Region" = "AFR",
    "South-East Asia Region" = "SEAR"
  )
}

figure_1_country_who_region <- function() {
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

load_figure_1_inputs <- function() {
  list(
    regional_incidence = readr::read_csv(
      model_path("data", "processed", "who_pertussis_region_incidence.csv"),
      show_col_types = FALSE
    ),
    pediatric_burden = read_table("lancet_baseline_pediatric_burden.csv"),
    intervention_summary = readr::read_csv(
      model_path("outputs", "summaries", "intervention_scenarios_summary.csv"),
      show_col_types = FALSE
    )
  )
}

prepare_figure_1_data <- function(inputs = load_figure_1_inputs()) {
  selected_regions <- figure_1_selected_regions()
  region_short <- figure_1_region_short_labels()

  regional_incidence <- inputs$regional_incidence %>%
    required_columns(
      c("region", "year", "reported_incidence_per_million"),
      "WHO regional incidence input"
    ) %>%
    mutate(reported_incidence_per_100k = reported_incidence_per_million / 10)

  regional_focus <- regional_incidence %>%
    filter(region == "Global" | region %in% selected_regions) %>%
    mutate(region = factor(region, levels = c("Global", selected_regions)))

  regional_focus_latest <- regional_focus %>%
    group_by(region) %>%
    filter(positive_rate(reported_incidence_per_100k)) %>%
    slice_max(year, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(region_label = region_short[as.character(region)])

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

  country_order <- pediatric_burden %>%
    arrange(desc(primary_cases_per_100k)) %>%
    mutate(country_label_text = format_country(country)) %>%
    pull(country_label_text)

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

  endpoint_spans <- baseline_endpoint_long %>%
    group_by(country, country_label_text, country_y_base) %>%
    summarise(
      min_rate_per_100k = min(rate_per_100k, na.rm = TRUE),
      max_rate_per_100k = max(rate_per_100k, na.rm = TRUE),
      .groups = "drop"
    )

  list(
    selected_regions = selected_regions,
    region_short = region_short,
    regional_focus = regional_focus,
    regional_focus_latest = regional_focus_latest,
    pediatric_burden = pediatric_burden,
    country_order = country_order,
    decision_map = decision_map,
    decision_medians = decision_medians,
    baseline_composition = baseline_composition,
    baseline_endpoint_long = baseline_endpoint_long,
    endpoint_spans = endpoint_spans
  )
}
