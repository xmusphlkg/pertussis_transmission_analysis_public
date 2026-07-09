## Figure 4 data preparation ---------------------------------------------------

figure_4_country_codes <- function() {
  codes <- country_codes
  codes[["United_Kingdom"]] <- "GBR"
  codes
}

figure_4_colours <- function() {
  list(
    case_benefit = manuscript_discrete_core[["blue"]],
    resistance = manuscript_discrete_core[["vermillion"]],
    preference_management = manuscript_discrete_core[["purple"]],
    preference_timeliness = manuscript_discrete_core[["deep_teal"]],
    implementation_guided = manuscript_discrete_core[["green"]],
    implementation_pep_restored = manuscript_discrete_core[["orange"]],
    implementation_lower_reach = manuscript_discrete_core[["grey"]]
  )
}

load_figure_4_vaccine_summary <- function() {
  read_model_table(model_path("outputs", "summaries", "vaccine_scenarios_summary")) %>%
    add_country_label() %>%
    mutate(
      scenario = factor(scenario, levels = vaccine_levels),
      scenario_label = factor(vaccine_labels[as.character(scenario)], levels = vaccine_labels[vaccine_levels])
    )
}

load_figure_4_inputs <- function() {
  veinf_summary_names <- c(
    "veinf_resistance_grid_summary.csv",
    "veinf_resistance_grid_low_seed_summary.csv"
  )

  list(
    decision_frontier = read_table("lancet_child_adolescent_decision_frontier.csv"),
    mechanism_summary = read_summary("resistance_mechanism_decomposition_summary.csv"),
    preference_summary = read_table("resistance_preference_weight_summary.csv"),
    preference_thresholds = read_table("resistance_preference_country_thresholds.csv"),
    implementation_summary = read_summary("treatment_implementation_sensitivity_summary.csv"),
    vaccine_summary = load_figure_4_vaccine_summary(),
    child_strategy_burden = read_table("lancet_child_adolescent_strategy_burden.csv"),
    veinf_grid_raw = bind_rows(
      lapply(
        veinf_summary_names[file.exists(model_path("outputs", "summaries", veinf_summary_names))],
        read_summary
      )
    ),
    veinf_threshold_comparators = read_summary("intervention_scenarios_summary.csv")
  )
}

prepare_figure_4_data <- function(inputs = load_figure_4_inputs()) {
  colours <- figure_4_colours()
  figure4_country_codes <- figure_4_country_codes()

  vaccine_profile <- vaccine_profile_spec("mechanism")
  vaccine_profile_labels <- vaccine_profile$labels
  vaccine_profile_axis_labels <- vaccine_profile$axis_labels
  vaccine_display_levels <- vaccine_profile$display_levels
  residual_display_levels <- vaccine_profile$residual_display_levels
  vaccine_legend_levels <- vaccine_profile$legend_levels
  vaccine_outcome_colours <- vaccine_profile$colours

  decision_frontier <- inputs$decision_frontier %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_label_text = format_country(country),
      country_code = unname(figure4_country_codes[country]),
      strategy = as.character(strategy),
      primary_case_reduction = as.numeric(primary_case_reduction),
      primary_cases_per_100k = as.numeric(primary_cases_per_100k),
      current_primary_cases_per_100k = as.numeric(current_primary_cases_per_100k),
      annualized_resistant_infections_per_100k = as.numeric(annualized_resistant_infections_per_100k),
      relative_reduction_resistant_infections = as.numeric(relative_reduction_resistant_infections),
      primary_case_rank_within_constraint = as.numeric(primary_case_rank_within_constraint)
    )

  current_resistance <- decision_frontier %>%
    filter(optimization_constraint == "program_plus_resistance", strategy == "current") %>%
    transmute(
      country,
      current_resistant_infections_per_100k = annualized_resistant_infections_per_100k,
      current_child_adolescent_cases_per_100k = current_primary_cases_per_100k
    )

  mechanism_order <- c(
    "baseline_full_mechanism",
    "no_resistant_importation",
    "equal_treatment_effect",
    "equal_pep_effect",
    "no_treatment_or_pep_differential",
    "fitness_cost"
  )

  mechanism_labels <- c(
    baseline_full_mechanism = "Baseline",
    no_resistant_importation = "Remove further resistant importation",
    equal_treatment_effect = "Equalise treatment benefit",
    equal_pep_effect = "Equalise resistant-strain PEP effectiveness",
    no_treatment_or_pep_differential = "Equalise treatment and PEP effects",
    fitness_cost = "Add resistant-strain fitness cost"
  )

  mechanism_plot_order <- c(
    "no_resistant_importation",
    "equal_treatment_effect",
    "equal_pep_effect",
    "no_treatment_or_pep_differential",
    "fitness_cost"
  )

  mechanism_axis_labels <- c(
    "Remove further resistant importation" = "Remove further\nresistant importation",
    "Equalise treatment benefit" = "Equalise treatment\nbenefit",
    "Equalise resistant-strain PEP effectiveness" = "Equalise resistant-strain\nPEP effectiveness",
    "Equalise treatment and PEP effects" = "Equalise treatment\nand PEP effects",
    "Add resistant-strain fitness cost" = "Add resistant-strain\nfitness cost"
  )

  mechanism_country <- inputs$mechanism_summary %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      scenario = mechanism_scenario,
      resistant_infections_per_100k = resistant_infections / total_population / analysis_years * 1e5,
      child_adolescent_cases_per_100k = annualized_child_adolescent_cases_per_100k
    )

  mechanism_data <- mechanism_country %>%
    group_by(mechanism_scenario) %>%
    summarise(
      median_resistant_infections_per_100k = median(resistant_infections_per_100k, na.rm = TRUE),
      q25_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.25),
      q75_resistant_infections_per_100k = interval_quantile(resistant_infections_per_100k, 0.75),
      median_child_adolescent_cases_per_100k = median(child_adolescent_cases_per_100k, na.rm = TRUE),
      q25_child_adolescent_cases_per_100k = interval_quantile(child_adolescent_cases_per_100k, 0.25),
      q75_child_adolescent_cases_per_100k = interval_quantile(child_adolescent_cases_per_100k, 0.75),
      median_end_resistant_fraction = median(resistant_fraction_end, na.rm = TRUE),
      resistant_importation = first(resistant_importation),
      treatment_differential = first(treatment_differential),
      pep_differential = first(pep_differential),
      fitness_R = first(fitness_R),
      interpretation = first(mechanism_interpretation),
      .groups = "drop"
    ) %>%
    mutate(
      scenario = factor(mechanism_scenario, levels = mechanism_order),
      mechanism_label = factor(
        mechanism_labels[as.character(scenario)],
        levels = c("Baseline", mechanism_labels[mechanism_plot_order])
      ),
      end_fraction_label = lancet_percent(median_end_resistant_fraction, accuracy = 1)
    ) %>%
    filter(!is.na(scenario), is.finite(median_resistant_infections_per_100k)) %>%
    arrange(scenario)

  mechanism_baseline <- mechanism_country %>%
    filter(scenario == "baseline_full_mechanism") %>%
    transmute(
      country,
      baseline_resistant_infections_per_100k = resistant_infections_per_100k,
      baseline_child_adolescent_cases_per_100k = child_adolescent_cases_per_100k
    )

  mechanism_delta <- mechanism_country %>%
    left_join(mechanism_baseline, by = "country") %>%
    filter(scenario != "baseline_full_mechanism") %>%
    mutate(
      mechanism_label = factor(
        mechanism_labels[scenario],
        levels = mechanism_labels[mechanism_plot_order]
      ),
      resistant_infection_reduction_vs_baseline = if_else(
        baseline_resistant_infections_per_100k > 0,
        1 - resistant_infections_per_100k / baseline_resistant_infections_per_100k,
        NA_real_
      ),
      child_adolescent_case_reduction_vs_baseline =
        1 - child_adolescent_cases_per_100k / pmax(baseline_child_adolescent_cases_per_100k, 1e-9)
    ) %>%
    pivot_longer(
      c(resistant_infection_reduction_vs_baseline, child_adolescent_case_reduction_vs_baseline),
      names_to = "metric",
      values_to = "reduction_vs_baseline"
    ) %>%
    mutate(
      metric = recode(
        metric,
        resistant_infection_reduction_vs_baseline = "Resistant infections",
        child_adolescent_case_reduction_vs_baseline = "Cases aged <18 years"
      ),
      metric = factor(metric, levels = c("Resistant infections", "Cases aged <18 years"))
    )

  mechanism_y_lookup <- setNames(
    seq_along(levels(mechanism_delta$mechanism_label)),
    levels(mechanism_delta$mechanism_label)
  )

  mechanism_metric_offsets <- c(
    "Cases aged <18 years" = -0.18,
    "Resistant infections" = 0.18
  )

  mechanism_delta <- mechanism_delta %>%
    mutate(
      mechanism_y = unname(mechanism_y_lookup[as.character(mechanism_label)]),
      mechanism_y_metric = mechanism_y + unname(mechanism_metric_offsets[as.character(metric)])
    )

  mechanism_delta_summary <- mechanism_delta %>%
    group_by(scenario, mechanism_label, metric) %>%
    summarise(
      mechanism_y = first(mechanism_y),
      mechanism_y_metric = first(mechanism_y_metric),
      median_reduction_vs_baseline = median(reduction_vs_baseline, na.rm = TRUE),
      q25_reduction_vs_baseline = interval_quantile(reduction_vs_baseline, 0.25),
      q75_reduction_vs_baseline = interval_quantile(reduction_vs_baseline, 0.75),
      .groups = "drop"
    )

  mechanism_y_breaks <- tibble::tibble(
    mechanism_label = names(mechanism_y_lookup),
    mechanism_y = unname(mechanism_y_lookup),
    mechanism_axis_label = unname(mechanism_axis_labels[mechanism_label])
  )

  mechanism_outcome_colours <- c(
    "Cases aged <18 years" = colours$case_benefit,
    "Resistant infections" = colours$resistance
  )

  mechanism_outcome_breaks <- levels(mechanism_delta$metric)
  mechanism_outcome_labels <- c(
    "Resistant infections" = "Resistant infections",
    "Cases aged <18 years" = "All <18 cases"
  )

  mechanism_point_legend <- tibble::tibble(
    reduction_vs_baseline = 0,
    mechanism_y_metric = mechanism_y_breaks$mechanism_y[[1]],
    point_summary = factor(
      c("Profiles", "Cross-profile median"),
      levels = c("Profiles", "Cross-profile median")
    )
  )

  mechanism_profile_points <- mechanism_delta %>%
    filter(is.finite(reduction_vs_baseline))

  resistance_management <- decision_frontier %>%
    filter(optimization_constraint == "program_plus_resistance", strategy == "resistance_guided_treatment") %>%
    select(
      country,
      country_label_text,
      country_code,
      primary_case_reduction,
      primary_cases_per_100k,
      relative_reduction_resistant_infections,
      remaining_resistant_infections_per_100k = annualized_resistant_infections_per_100k
    ) %>%
    left_join(current_resistance, by = "country") %>%
    mutate(
      resistance_reduction_estimable = is.finite(relative_reduction_resistant_infections),
      resistant_reduction_plot = if_else(resistance_reduction_estimable, relative_reduction_resistant_infections, 0),
      current_resistant_infections_plot = pmax(current_resistant_infections_per_100k, 1)
    )

  p4b_label_data <- resistance_management %>%
    mutate(
      label_nudge_x = case_when(
        country_code %in% c("BRA", "GBR", "NZL", "SWE", "USA") ~ -0.030,
        country_code %in% c("AUS", "CHN", "THA", "ZAF") ~ 0.030,
        TRUE ~ 0.005
      ),
      label_nudge_y = case_when(
        country_code == "AUS" ~ -0.020,
        country_code == "BRA" ~ 0.035,
        country_code == "CHN" ~ 0.025,
        country_code == "GBR" ~ 0.028,
        country_code == "JPN" ~ 0.024,
        country_code == "NZL" ~ -0.024,
        country_code == "THA" ~ 0.044,
        country_code == "USA" ~ -0.038,
        country_code == "ZAF" ~ -0.020,
        TRUE ~ 0
      )
    )

  pal_breaks <- c(1, 100, 1000)
  pal_limits <- c(1, 3500)

  preference_summary <- inputs$preference_summary %>%
    filter(strategy %in% c("resistance_guided_treatment", "timeliness_only")) %>%
    mutate(
      strategy = as.character(strategy),
      strategy_label = recode(
        strategy,
        resistance_guided_treatment = "Resistance-guided management",
        timeliness_only = "Routine timeliness"
      ),
      strategy_label = factor(strategy_label, levels = c("Resistance-guided management", "Routine timeliness")),
      resistance_weight_lambda = as.numeric(resistance_weight_lambda),
      countries_preferred = as.numeric(countries_preferred)
    )

  preference_thresholds <- inputs$preference_thresholds %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_code = unname(figure4_country_codes[country]),
      minimum_lambda_for_resistance_guided_preferred =
        as.numeric(minimum_lambda_for_resistance_guided_preferred),
      resistance_guided_resistant_infection_reduction =
        as.numeric(resistance_guided_resistant_infection_reduction),
      threshold_label = if_else(minimum_lambda_for_resistance_guided_preferred > 0, country_code, NA_character_)
    )

  preference_colours <- c(
    "Resistance-guided management" = colours$preference_management,
    "Routine timeliness" = colours$preference_timeliness
  )

  p4c_curve_label_targets <- tibble::tibble(
    strategy_label = factor(
      c("Resistance-guided management", "Routine timeliness"),
      levels = levels(preference_summary$strategy_label)
    ),
    label_lambda = c(0.64, 0.20),
    curve_label = c("Resistance-guided\nmanagement", "Routine\ntimeliness"),
    label_x = c(0.68, 0.20),
    label_y_offset = c(-0.52, 0.68)
  )

  p4c_curve_label_data <- preference_summary %>%
    inner_join(
      p4c_curve_label_targets,
      by = c("strategy_label", "resistance_weight_lambda" = "label_lambda")
    ) %>%
    mutate(label_y = countries_preferred + label_y_offset)

  p4c_threshold_tick_data <- preference_thresholds %>%
    mutate(
      threshold_tick_ymin = case_when(
        !is.na(threshold_label) ~ -0.94,
        TRUE ~ -0.42
      )
    )

  p4c_threshold_label_data <- preference_thresholds %>%
    filter(!is.na(threshold_label)) %>%
    mutate(
      threshold_label_y = case_when(
        threshold_label == "GBR" ~ -0.38,
        threshold_label == "NZL" ~ -0.82,
        TRUE ~ -0.62
      )
    )

  implementation_summary <- inputs$implementation_summary

  near_term_current <- implementation_summary %>%
    filter(scenario == "current_near_term") %>%
    transmute(
      country,
      current_child_adolescent_cases = total_child_adolescent_cases,
      current_resistant_infections = resistant_infections
    )

  implementation_country <- implementation_summary %>%
    left_join(near_term_current, by = "country") %>%
    mutate(
      raw_child_adolescent_case_reduction = 1 - total_child_adolescent_cases / current_child_adolescent_cases,
      child_adolescent_case_reduction = bounded_proportional_response(
        current_child_adolescent_cases,
        total_child_adolescent_cases
      ),
      raw_resistant_infection_reduction = if_else(
        current_resistant_infections > 0,
        1 - resistant_infections / current_resistant_infections,
        NA_real_
      ),
      resistant_infection_reduction = if_else(
        current_resistant_infections > 0,
        bounded_proportional_response(current_resistant_infections, resistant_infections),
        NA_real_
      )
    )

  implementation_sensitivity <- implementation_country %>%
    group_by(scenario, implementation_uptake, pep_restored, pep_coverage_multiplier, implementation_note) %>%
    summarise(
      median_child_adolescent_case_reduction = median(child_adolescent_case_reduction, na.rm = TRUE),
      q25_child_adolescent_case_reduction = interval_quantile(child_adolescent_case_reduction, 0.25),
      q75_child_adolescent_case_reduction = interval_quantile(child_adolescent_case_reduction, 0.75),
      median_resistant_infection_reduction = median(resistant_infection_reduction, na.rm = TRUE),
      q25_resistant_infection_reduction = interval_quantile(resistant_infection_reduction, 0.25),
      q75_resistant_infection_reduction = interval_quantile(resistant_infection_reduction, 0.75),
      .groups = "drop"
    ) %>%
    mutate(
      pep_assumption = case_when(
        pep_restored == "yes" ~ "Treatment + restored PEP effect",
        TRUE ~ "Guided treatment only"
      ),
      reach_status = if_else(pep_coverage_multiplier < 1, "Lower PEP reach", "Baseline PEP reach"),
      implementation_group = case_when(
        reach_status == "Lower PEP reach" ~ "Lower reach",
        pep_assumption == "Treatment + restored PEP effect" ~ "+ restored PEP",
        TRUE ~ "Guided only"
      ),
      uptake_offset = case_when(
        scenario == "current_near_term" ~ 0,
        pep_assumption == "Guided treatment only" & reach_status == "Baseline PEP reach" ~ -0.018,
        pep_assumption == "Treatment + restored PEP effect" & reach_status == "Baseline PEP reach" ~ 0.018,
        reach_status == "Lower PEP reach" ~ 0.046,
        TRUE ~ 0
      ),
      uptake_plot = implementation_uptake + uptake_offset
    )

  reduction_axis_floor <- -1.0
  reduction_axis_ceiling <- 0.85

  implementation_plot <- bind_rows(
    implementation_sensitivity %>%
      transmute(
        scenario,
        uptake = implementation_uptake,
        uptake_offset,
        uptake_plot,
        pep_assumption,
        reach_status,
        implementation_group,
        metric = "Cases aged <18 years",
        median_reduction = median_child_adolescent_case_reduction,
        q25_reduction = q25_child_adolescent_case_reduction,
        q75_reduction = q75_child_adolescent_case_reduction,
        implementation_note
      ),
    implementation_sensitivity %>%
      transmute(
        scenario,
        uptake = implementation_uptake,
        uptake_offset,
        uptake_plot,
        pep_assumption,
        reach_status,
        implementation_group,
        metric = "Resistant infections",
        median_reduction = median_resistant_infection_reduction,
        q25_reduction = q25_resistant_infection_reduction,
        q75_reduction = q75_resistant_infection_reduction,
        implementation_note
      )
  ) %>%
    mutate(
      metric = factor(metric, levels = c("Cases aged <18 years", "Resistant infections")),
      pep_assumption = factor(pep_assumption, levels = c("Guided treatment only", "Treatment + restored PEP effect")),
      reach_status = factor(reach_status, levels = c("Baseline PEP reach", "Lower PEP reach")),
      implementation_group = factor(
        implementation_group,
        levels = c("Guided only", "+ restored PEP", "Lower reach")
      ),
      q25_reduction_plot = pmax(q25_reduction, reduction_axis_floor),
      q75_reduction_plot = pmin(q75_reduction, reduction_axis_ceiling),
      q25_truncated = q25_reduction < reduction_axis_floor,
      q75_truncated = q75_reduction > reduction_axis_ceiling,
      effect_measure = "bounded_proportional_response"
    )

  current_vaccine_reference <- inputs$vaccine_summary %>%
    filter(as.character(scenario) == "symptom_protective") %>%
    transmute(
      country = stringr::str_replace_all(country, " ", "_"),
      current_infections_per_100k = as.numeric(annualized_infections_per_100k),
      current_primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k)
    )

  vaccine_mechanism_outcomes <- inputs$vaccine_summary %>%
    filter(as.character(scenario) %in% vaccine_display_levels) %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      scenario = as.character(scenario)
    ) %>%
    left_join(current_vaccine_reference, by = "country") %>%
    mutate(
      country_code = unname(figure4_country_codes[country]),
      scenario_label = factor(unname(vaccine_profile_labels[scenario]), levels = vaccine_profile$display_labels),
      primary_cases_per_100k = as.numeric(annualized_child_adolescent_cases_per_100k),
      primary_case_reduction = 1 - primary_cases_per_100k / pmax(current_primary_cases_per_100k, 1e-9),
      relative_reduction_total_infections = 1 - as.numeric(annualized_infections_per_100k) / pmax(current_infections_per_100k, 1e-9)
    ) %>%
    filter(!is.na(primary_case_reduction), !is.na(relative_reduction_total_infections), !is.na(primary_cases_per_100k))

  combined_stress_outcomes <- inputs$child_strategy_burden %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      country_code = unname(figure4_country_codes[country]),
      strategy = as.character(strategy),
      primary_cases_per_100k = as.numeric(primary_cases_per_100k)
    ) %>%
    filter(strategy == "combined_strategy") %>%
    transmute(
      country,
      country_code,
      scenario = strategy,
      scenario_label = "Combined future upper-bound scenario",
      primary_cases_per_100k,
      outcome_type = "Upper-bound scenario"
    )

  vaccine_residual <- vaccine_mechanism_outcomes %>%
    transmute(
      country,
      country_code,
      scenario,
      scenario_label = as.character(scenario_label),
      primary_cases_per_100k,
      outcome_type = "Vaccine mechanism"
    ) %>%
    bind_rows(combined_stress_outcomes) %>%
    mutate(
      scenario_label = factor(scenario_label, levels = rev(residual_display_levels)),
      country_label_text = format_country(country),
      outcome_type = factor(outcome_type, levels = c("Vaccine mechanism", "Upper-bound scenario"))
    )

  vaccine_burden_summary <- vaccine_residual %>%
    group_by(scenario_label) %>%
    summarise(
      median = median(primary_cases_per_100k, na.rm = TRUE),
      q25 = interval_quantile(primary_cases_per_100k, 0.25),
      q75 = interval_quantile(primary_cases_per_100k, 0.75),
      q025 = interval_quantile(primary_cases_per_100k, 0.025),
      q975 = interval_quantile(primary_cases_per_100k, 0.975),
      min_primary_cases_per_100k = min(primary_cases_per_100k, na.rm = TRUE),
      max_primary_cases_per_100k = max(primary_cases_per_100k, na.rm = TRUE),
      .groups = "drop"
    )

  veinf_grid_raw <- inputs$veinf_grid_raw %>%
    mutate(
      country = stringr::str_replace_all(country, " ", "_"),
      starting_resistant_fraction = round(as.numeric(grid_resistance_prevalence), 2),
      infectiousness_reduction = round(as.numeric(grid_VE_inf), 2),
      infant_cases_per_100k = as.numeric(annualized_infant_cases_per_100k)
    )

  veinf_plot_resistance_levels <- c(0, 0.01, 0.03, 0.05)
  veinf_resistance_levels <- veinf_plot_resistance_levels[
    veinf_plot_resistance_levels %in% sort(unique(veinf_grid_raw$starting_resistant_fraction))
  ]

  threshold_comparator_labels <- c(
    "Infant-exposure composite",
    "Resistance-guided management"
  )

  veinf_threshold_grid <- veinf_grid_raw %>%
    filter(
      starting_resistant_fraction %in% veinf_resistance_levels,
      infectiousness_reduction >= 0,
      infectiousness_reduction <= 0.6
    )

  veinf_threshold_comparators <- inputs$veinf_threshold_comparators %>%
    filter(as.character(scenario) %in% c("maternal_immunization", "resistance_guided_treatment")) %>%
    transmute(
      country = stringr::str_replace_all(country, " ", "_"),
      comparator = recode(
        as.character(scenario),
        maternal_immunization = "infant_exposure_reduction_strategy",
        resistance_guided_treatment = "resistance_guided_treatment"
      ),
      comparator_label = recode(
        comparator,
        infant_exposure_reduction_strategy = "Infant-exposure composite",
        resistance_guided_treatment = "Resistance-guided management"
      ),
      comparator_infant_cases_per_100k = as.numeric(annualized_infant_cases_per_100k)
    )

  veinf_country_thresholds <- veinf_threshold_grid %>%
    inner_join(veinf_threshold_comparators, by = "country", relationship = "many-to-many") %>%
    group_by(
      country,
      starting_resistant_fraction,
      comparator,
      comparator_label,
      comparator_infant_cases_per_100k
    ) %>%
    summarise(
      minimum_infectiousness_reduction = {
        eligible <- infectiousness_reduction[infant_cases_per_100k <= comparator_infant_cases_per_100k]
        if (length(eligible) == 0) NA_real_ else min(eligible, na.rm = TRUE)
      },
      .groups = "drop"
    ) %>%
    mutate(
      resistance_label = factor(
        paste0(lancet_percent(starting_resistant_fraction, accuracy = 1), " resistant-strain seeding"),
        levels = paste0(lancet_percent(veinf_resistance_levels, accuracy = 1), " resistant-strain seeding")
      ),
      comparator_label = factor(comparator_label, levels = threshold_comparator_labels),
      reached_comparator = is.finite(minimum_infectiousness_reduction)
    )

  veinf_threshold_summary <- veinf_country_thresholds %>%
    group_by(starting_resistant_fraction, resistance_label, comparator, comparator_label) %>%
    summarise(
      median_minimum_infectiousness_reduction = median(minimum_infectiousness_reduction, na.rm = TRUE),
      q25_minimum_infectiousness_reduction = interval_quantile(minimum_infectiousness_reduction, 0.25),
      q75_minimum_infectiousness_reduction = interval_quantile(minimum_infectiousness_reduction, 0.75),
      min_minimum_infectiousness_reduction = suppressWarnings(min(minimum_infectiousness_reduction, na.rm = TRUE)),
      max_minimum_infectiousness_reduction = suppressWarnings(max(minimum_infectiousness_reduction, na.rm = TRUE)),
      countries_reaching_comparator = sum(reached_comparator, na.rm = TRUE),
      countries_evaluated = n(),
      .groups = "drop"
    ) %>%
    mutate(
      min_minimum_infectiousness_reduction = if_else(
        is.infinite(min_minimum_infectiousness_reduction),
        NA_real_,
        min_minimum_infectiousness_reduction
      ),
      max_minimum_infectiousness_reduction = if_else(
        is.infinite(max_minimum_infectiousness_reduction),
        NA_real_,
        max_minimum_infectiousness_reduction
      ),
      countries_label = paste0(countries_reaching_comparator, "/", countries_evaluated, " profiles"),
      threshold_basis = "VE_inf-only grid; VE_sus and VE_dur held at the grid baseline.",
      label_vjust = case_when(
        comparator == "infant_exposure_reduction_strategy" & starting_resistant_fraction > 0 ~ 1.55,
        TRUE ~ -0.62
      )
    )

  veinf_threshold_source <- veinf_country_thresholds %>%
    left_join(
      veinf_threshold_summary %>%
        select(
          starting_resistant_fraction,
          comparator,
          median_minimum_infectiousness_reduction,
          q25_minimum_infectiousness_reduction,
          q75_minimum_infectiousness_reduction,
          min_minimum_infectiousness_reduction,
          max_minimum_infectiousness_reduction,
          countries_reaching_comparator,
          countries_evaluated,
          threshold_basis
        ),
      by = c("starting_resistant_fraction", "comparator")
    ) %>%
    arrange(starting_resistant_fraction, comparator, country) %>%
    transmute(
      country,
      starting_resistant_fraction,
      resistance_label = as.character(resistance_label),
      comparator,
      comparator_label = as.character(comparator_label),
      comparator_infant_cases_per_100k,
      minimum_infectiousness_reduction,
      reached_comparator,
      median_minimum_infectiousness_reduction,
      q25_minimum_infectiousness_reduction,
      q75_minimum_infectiousness_reduction,
      min_minimum_infectiousness_reduction,
      max_minimum_infectiousness_reduction,
      countries_reaching_comparator,
      countries_evaluated,
      threshold_basis
    )

  veinf_attainment_country <- veinf_country_thresholds %>%
    select(
      country,
      starting_resistant_fraction,
      resistance_label,
      comparator,
      comparator_label,
      comparator_infant_cases_per_100k,
      minimum_infectiousness_reduction,
      reached_comparator
    ) %>%
    mutate(join_key = 1L) %>%
    inner_join(
      tibble(
        join_key = 1L,
        infectiousness_reduction = sort(unique(veinf_threshold_grid$infectiousness_reduction))
      ),
      by = "join_key",
      relationship = "many-to-many"
    ) %>%
    select(-join_key) %>%
    mutate(
      matched_comparator = is.finite(minimum_infectiousness_reduction) &
        minimum_infectiousness_reduction <= infectiousness_reduction
    )

  veinf_attainment_summary <- veinf_attainment_country %>%
    group_by(
      comparator,
      comparator_label,
      starting_resistant_fraction,
      resistance_label,
      infectiousness_reduction
    ) %>%
    summarise(
      profiles_matching_comparator = sum(matched_comparator, na.rm = TRUE),
      countries_evaluated = n(),
      .groups = "drop"
    ) %>%
    mutate(
      profiles_label = paste0(profiles_matching_comparator, "/", countries_evaluated, " reached")
    )

  veinf_attainment_plot_summary <- veinf_attainment_summary %>%
    filter(starting_resistant_fraction %in% veinf_resistance_levels) %>%
    mutate(
      seed_label = factor(
        lancet_percent(starting_resistant_fraction, accuracy = 1),
        levels = lancet_percent(veinf_resistance_levels, accuracy = 1)
      )
    )

  veinf_attainment_source <- veinf_attainment_country %>%
    left_join(
      veinf_attainment_summary %>%
        select(
          comparator,
          starting_resistant_fraction,
          infectiousness_reduction,
          profiles_matching_comparator,
          countries_evaluated
        ),
      by = c("comparator", "starting_resistant_fraction", "infectiousness_reduction")
    ) %>%
    arrange(comparator, starting_resistant_fraction, infectiousness_reduction, country) %>%
    transmute(
      country,
      starting_resistant_fraction,
      resistance_label = as.character(resistance_label),
      comparator,
      comparator_label = as.character(comparator_label),
      infectiousness_reduction,
      comparator_infant_cases_per_100k,
      minimum_infectiousness_reduction,
      reached_comparator,
      matched_comparator,
      profiles_matching_comparator,
      countries_evaluated
    )

  veinf_seed_colours <- c(
    "0%" = manuscript_colour("black"),
    "1%" = okabe_ito[[6]],
    "3%" = okabe_ito[[2]],
    "5%" = okabe_ito[[8]]
  )

  veinf_seed_linetypes <- c(
    "0%" = "solid",
    "1%" = "longdash",
    "3%" = "dotdash",
    "5%" = "dashed"
  )

  veinf_seed_shapes <- c(
    "0%" = 16,
    "1%" = 17,
    "3%" = 15,
    "5%" = 18
  )

  list(
    colours = colours,
    country_codes = figure4_country_codes,
    vaccine_profile = vaccine_profile,
    vaccine_profile_labels = vaccine_profile_labels,
    vaccine_profile_axis_labels = vaccine_profile_axis_labels,
    vaccine_display_levels = vaccine_display_levels,
    residual_display_levels = residual_display_levels,
    vaccine_legend_levels = vaccine_legend_levels,
    vaccine_outcome_colours = vaccine_outcome_colours,
    decision_frontier = decision_frontier,
    current_resistance = current_resistance,
    mechanism_order = mechanism_order,
    mechanism_labels = mechanism_labels,
    mechanism_plot_order = mechanism_plot_order,
    mechanism_axis_labels = mechanism_axis_labels,
    mechanism_country = mechanism_country,
    mechanism_data = mechanism_data,
    mechanism_delta = mechanism_delta,
    mechanism_delta_summary = mechanism_delta_summary,
    mechanism_y_breaks = mechanism_y_breaks,
    mechanism_outcome_colours = mechanism_outcome_colours,
    mechanism_outcome_breaks = mechanism_outcome_breaks,
    mechanism_outcome_labels = mechanism_outcome_labels,
    mechanism_point_legend = mechanism_point_legend,
    mechanism_profile_points = mechanism_profile_points,
    resistance_management = resistance_management,
    p4b_label_data = p4b_label_data,
    pal_breaks = pal_breaks,
    pal_limits = pal_limits,
    preference_summary = preference_summary,
    preference_thresholds = preference_thresholds,
    preference_colours = preference_colours,
    p4c_curve_label_data = p4c_curve_label_data,
    p4c_threshold_tick_data = p4c_threshold_tick_data,
    p4c_threshold_label_data = p4c_threshold_label_data,
    implementation_summary = implementation_summary,
    implementation_sensitivity = implementation_sensitivity,
    implementation_plot = implementation_plot,
    reduction_axis_floor = reduction_axis_floor,
    reduction_axis_ceiling = reduction_axis_ceiling,
    vaccine_residual = vaccine_residual,
    vaccine_burden_summary = vaccine_burden_summary,
    veinf_threshold_source = veinf_threshold_source,
    veinf_attainment_source = veinf_attainment_source,
    veinf_attainment_plot_summary = veinf_attainment_plot_summary,
    veinf_seed_colours = veinf_seed_colours,
    veinf_seed_linetypes = veinf_seed_linetypes,
    veinf_seed_shapes = veinf_seed_shapes
  )
}
