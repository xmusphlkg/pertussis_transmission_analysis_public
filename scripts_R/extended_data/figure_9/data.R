## Extended Data Figure 9 data preparation ------------------------------------

extended_data_figure_9_strategy_labels <- function() {
  c(
    current = "Current",
    higher_child_coverage = "Nominal coverage floor",
    adolescent_booster = "Adolescent booster",
    pregnancy_tdap_scaleup = "Pregnancy Tdap",
    cocooning_adjunct = "Close-contact adjunct",
    maternal_immunization = "Infant-exposure composite",
    targeted_pep_high_risk = "Targeted PEP",
    resistance_guided_treatment = "Resistance-guided care",
    next_generation_vaccine = "High transmission blocking",
    combined_strategy = "Combined stress test"
  )
}

extended_data_figure_9_class_labels <- function() {
  c(
    combined_stress_test_package = "Combined stress test",
    high_transmission_blocking_vaccine_target = "High transmission blocking",
    infant_protection_and_exposure_reduction = "Infant protection/exposure",
    management_modifiers = "Management modifiers",
    current_practice = "Current practice",
    routine_program_marginal_levers = "Routine marginal strategies"
  )
}

extended_data_figure_9_strategy_order <- function() {
  c(
    "combined_strategy",
    "next_generation_vaccine",
    "maternal_immunization",
    "resistance_guided_treatment",
    "cocooning_adjunct",
    "pregnancy_tdap_scaleup",
    "targeted_pep_high_risk",
    "adolescent_booster",
    "current",
    "higher_child_coverage"
  )
}

extended_data_figure_9_programme_strategy_order <- function() {
  c(
    "timeliness_only",
    "maternal_immunization",
    "cocooning_adjunct",
    "targeted_pep_high_risk",
    "pregnancy_tdap_scaleup",
    "adolescent_booster"
  )
}

extended_data_figure_9_programme_strategy_labels <- function() {
  c(
    timeliness_only = "Routine schedule\ntimeliness",
    maternal_immunization = "Infant-exposure\ncomposite",
    cocooning_adjunct = "Close-contact\nadult adjuncts",
    targeted_pep_high_risk = "Targeted high-risk\nPEP",
    pregnancy_tdap_scaleup = "Pregnancy Tdap\nscale-up",
    adolescent_booster = "Adolescent booster\nscale-up"
  )
}

load_extended_data_figure_9_inputs <- function() {
  list(
    routine = read_csv_local("outputs", "tables", "routine_timeliness_sensitivity.csv"),
    horizon = read_csv_local("outputs", "tables", "intervention_horizon_rank_summary.csv"),
    age_split = read_csv_local("outputs", "tables", "infant_age_split_horizon_sensitivity.csv"),
    stability = read_csv_local("outputs", "tables", "intervention_rank_stability_diagnostics.csv"),
    psa = read_csv_local(
      "outputs", "tables", "joint_psa_under18_programme_rank_acceptability.csv"
    ),
    age_pattern = read_csv_local("outputs", "tables", "age_pattern_scenario_ordering_sensitivity.csv"),
    age_pattern_programme = read_csv_local("outputs", "tables", "lancet_age_pattern_weighted_strategy_summary.csv")
  )
}

prepare_extended_data_figure_9_data <- function(inputs = load_extended_data_figure_9_inputs()) {
  strategy_labels <- extended_data_figure_9_strategy_labels()
  class_labels <- extended_data_figure_9_class_labels()
  strategy_order <- extended_data_figure_9_strategy_order()
  programme_strategy_order <- extended_data_figure_9_programme_strategy_order()
  programme_strategy_labels <- extended_data_figure_9_programme_strategy_labels()

  routine <- inputs$routine %>%
    mutate(
      strategy_label = recode(
        strategy,
        current = "Current",
        coverage_floor_only = "Nominal coverage floor",
        timeliness_only = "Timeliness only",
        coverage_floor_plus_timeliness = "Coverage + timeliness"
      ),
      strategy_label = factor(
        strategy_label,
        levels = c("Current", "Nominal coverage floor", "Timeliness only", "Coverage + timeliness")
      )
    )

  horizon <- inputs$horizon %>%
    mutate(
      scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
      analysis_window = recode(
        analysis_window,
        first_5_years = "First 5 years",
        first_10_years = "First 10 years",
        first_15_years = "First 15 years",
        excluding_initial_5_years = "After first 5 years",
        full_horizon = "Full horizon"
      )
    )

  age_summary <- inputs$age_split %>%
    filter(analysis_window == "full_horizon", scenario %in% strategy_order) %>%
    group_by(scenario, age_group) %>%
    summarise(median_reduction = median(relative_reduction_infant_cases, na.rm = TRUE), .groups = "drop") %>%
    mutate(
      scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
      age_group = recode(age_group, infant_0_2m = "0–2 mo", infant_3_11m = "3–11 mo")
    )

  stability <- inputs$stability %>%
    filter(scenario %in% strategy_order) %>%
    mutate(
      scenario_label = factor(recode(scenario, !!!strategy_labels), levels = rev(strategy_labels[strategy_order])),
      window_top2 = analysis_window_cells_ranked_top_two / analysis_window_cells,
      infant_top2 = infant_age_window_cells_ranked_top_two / infant_age_window_cells
    ) %>%
    select(scenario_label, window_top2, infant_top2, median_infant_age_window_reduction) %>%
    pivot_longer(c(window_top2, infant_top2), names_to = "diagnostic", values_to = "share") %>%
    mutate(diagnostic = recode(diagnostic, window_top2 = "Window cells top 2", infant_top2 = "Infant-age cells top 2"))

  psa <- inputs$psa %>%
    filter(country == "All_countries_pooled", rank == 1, strategy %in% programme_strategy_order) %>%
    distinct(strategy, .keep_all = TRUE) %>%
    mutate(
      strategy_label = factor(
        programme_strategy_labels[strategy],
        levels = rev(programme_strategy_labels[programme_strategy_order])
      )
    )

  age_pattern <- inputs$age_pattern %>%
    filter(ordering_basis %in% c("all_profiles_unweighted", "external_age_pattern_weighted", "external_age_pattern_pass_filter")) %>%
    mutate(
      ordering_basis = recode(
        ordering_basis,
        all_profiles_unweighted = "All profiles",
        external_age_pattern_weighted = "Age-pattern weighted",
        external_age_pattern_pass_filter = "Pass filter"
      ),
      scenario_class_label = factor(recode(scenario_class, !!!class_labels), levels = rev(class_labels))
    )

  age_pattern_programme <- inputs$age_pattern_programme %>%
    filter(
      strategy %in% programme_strategy_order,
      ordering_basis %in% c(
        "all_profiles_unweighted",
        "age_data_profiles_unweighted",
        "age_pattern_weighted"
      )
    ) %>%
    mutate(
      strategy_label = stringr::str_replace_all(as.character(programme_strategy_labels[strategy]), "\n", " "),
      strategy_label_plot = factor(
        programme_strategy_labels[strategy],
        levels = rev(programme_strategy_labels[programme_strategy_order])
      ),
      weighted_median_primary_case_reduction = as.numeric(weighted_median_primary_case_reduction),
      weighted_median_primary_cases_per_100k = as.numeric(weighted_median_primary_cases_per_100k),
      weighted_iqr_primary_case_reduction = as.character(weighted_iqr_primary_case_reduction),
      weighted_iqr_low_primary_case_reduction = as.numeric(stringr::str_match(
        weighted_iqr_primary_case_reduction,
        "^\\s*([-0-9.eE]+)\\s+to\\s+([-0-9.eE]+)\\s*$"
      )[, 2]),
      weighted_iqr_high_primary_case_reduction = as.numeric(stringr::str_match(
        weighted_iqr_primary_case_reduction,
        "^\\s*([-0-9.eE]+)\\s+to\\s+([-0-9.eE]+)\\s*$"
      )[, 3]),
      ordering_basis_label = recode(
        ordering_basis,
        all_profiles_unweighted = "All profiles",
        age_data_profiles_unweighted = "Profiles with age data",
        age_pattern_weighted = "Age-pattern weighted"
      ),
      ordering_basis_label = factor(
        ordering_basis_label,
        levels = c("All profiles", "Profiles with age data", "Age-pattern weighted")
      )
    )

  list(
    routine = routine,
    horizon = horizon,
    age_summary = age_summary,
    stability = stability,
    psa = psa,
    age_pattern = age_pattern,
    age_pattern_programme = age_pattern_programme
  )
}
