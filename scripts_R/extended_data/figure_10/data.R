## Extended Data Figure 10 data preparation -----------------------------------

load_extended_data_figure_10_inputs <- function() {
  list(
    policy = read_csv_local("outputs", "tables", "resistance_management_policy_decomposition.csv"),
    preference = read_csv_local("outputs", "tables", "resistance_preference_weight_summary.csv"),
    thresholds = read_csv_local("outputs", "tables", "veinf_comparator_thresholds.csv")
  )
}

prepare_extended_data_figure_10_data <- function(inputs = load_extended_data_figure_10_inputs()) {
  mechanism_levels <- c(
    "Baseline resistant-strain mechanism",
    "Treatment differential removed",
    "PEP differential removed",
    "Treatment and PEP differentials removed",
    "No ongoing resistant importation",
    "Fitness-cost stress test"
  )

  mechanism <- inputs$policy %>%
    filter(analysis_layer == "Long-horizon mechanism decomposition") %>%
    mutate(
      policy_read = factor(policy_read, levels = rev(mechanism_levels)),
      policy_short = str_wrap(as.character(policy_read), width = 22)
    )

  implementation_bounds <- parse_range_bounds(inputs$policy$iqr_infant_case_reduction_vs_current)
  implementation <- inputs$policy %>%
    bind_cols(implementation_bounds) %>%
    filter(analysis_layer == "Near-term implementation sensitivity") %>%
    mutate(
      policy_read = str_squish(policy_read),
      policy_short = recode(
        policy_read,
        "Current near-term management" = "Current",
        "25% testing/treatment uptake plus assumed PEP improvement" = "25% uptake + improved PEP",
        "50% testing/treatment uptake plus assumed PEP improvement" = "50% uptake + improved PEP",
        "75% testing/treatment uptake plus assumed PEP improvement" = "75% uptake + improved PEP",
        "100% testing/treatment uptake plus assumed PEP improvement" = "100% uptake + improved PEP",
        "50% testing/treatment uptake only" = "50% uptake only",
        "100% testing/treatment uptake only" = "100% uptake only",
        "50% uptake plus assumed PEP improvement with lower PEP reach" = "50% uptake + improved PEP, lower reach"
      ),
      policy_short = fct_reorder(policy_short, median_infant_case_reduction_vs_current),
      pep_status = if_else(str_detect(pep_component, "improved"), "PEP improved", "Baseline PEP")
    )

  preference <- inputs$preference %>%
    filter(strategy %in% c("timeliness_only", "maternal_immunization", "adolescent_booster", "resistance_guided_treatment")) %>%
    mutate(
      strategy_label = recode(
        strategy,
        timeliness_only = "Timeliness",
        maternal_immunization = "Infant exposure",
        adolescent_booster = "Adolescent",
        resistance_guided_treatment = "Resistance mgmt"
      ),
      strategy_label = factor(
        strategy_label,
        levels = c("Timeliness", "Infant exposure", "Adolescent", "Resistance mgmt")
      )
    )

  thresholds <- inputs$thresholds %>%
    filter(!is.na(median_minimum_VE_inf)) %>%
    mutate(
      comparator = recode(
        comparator,
        infant_exposure_reduction_strategy = "Infant-exposure strategy",
        resistance_guided_treatment = "Resistance-guided care",
        `25% reduction vs VE_inf_0.20` = "25% infant-case reduction",
        `50% reduction vs VE_inf_0.20` = "50% infant-case reduction",
        `75% reduction vs VE_inf_0.20` = "75% infant-case reduction"
      ),
      comparator = factor(
        comparator,
        levels = c(
          "Resistance-guided care", "Infant-exposure strategy",
          "25% infant-case reduction", "50% infant-case reduction",
          "75% infant-case reduction"
        )
      ),
      resistance_prevalence_label = scales::percent(resistance_prevalence, accuracy = 1),
      countries_label = paste0(countries_reaching_comparator, "/", countries_evaluated)
    )

  list(
    mechanism = mechanism,
    implementation = implementation,
    preference = preference,
    thresholds = thresholds,
    threshold_y_upper = max(0.62, max(thresholds$median_minimum_VE_inf, na.rm = TRUE) * 1.05)
  )
}
