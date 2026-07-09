## Extended Data Figure 11 data preparation -----------------------------------

extended_data_figure_11_strategy_labels <- function() {
  c(
    current = "Current",
    maternal_immunization = "Infant-exposure strategy",
    maternal_direct_antibody_only = "Direct antibody only",
    pregnancy_tdap_scaleup = "Pregnancy Tdap",
    cocooning_adjunct = "Close-contact adjunct",
    adolescent_booster = "Adolescent booster",
    targeted_pep_high_risk = "Targeted PEP",
    resistance_guided_treatment = "Resistance-guided care",
    next_generation_vaccine = "High transmission blocking",
    combined_strategy = "Combined stress test",
    higher_child_coverage = "Nominal coverage floor"
  )
}

extended_data_figure_11_portfolio_labels <- function() {
  c(
    current = "Current",
    routine_timeliness = "Timeliness",
    timeliness_pregnancy_tdap = "Timeliness +\npregnancy Tdap",
    timeliness_targeted_pep = "Timeliness +\ntargeted PEP",
    infant_exposure = "Infant exposure",
    timeliness_infant_exposure = "Timeliness +\ninfant exposure",
    timeliness_infant_exposure_targeted_pep = "Timeliness + exposure +\ntargeted PEP",
    routine_timeliness_resistance_guided = "Timeliness +\nresistance mgmt",
    infant_exposure_resistance_guided = "Exposure +\nresistance mgmt",
    timeliness_infant_exposure_resistance_guided = "Timeliness + exposure +\nresistance mgmt",
    timeliness_infant_exposure_targeted_pep_resistance_guided = "Timeliness + exposure +\nPEP + resistance"
  )
}

load_extended_data_figure_11_inputs <- function() {
  list(
    portfolio = read_csv_local("outputs", "tables", "program_portfolio_factorial_summary.csv"),
    contact = read_csv_local("outputs", "tables", "infant_contact_sensitivity.csv"),
    maternal = read_csv_local("outputs", "tables", "maternal_duration_sensitivity.csv"),
    temporal = read_csv_local("outputs", "tables", "temporal_assumption_sensitivity.csv"),
    event_scale = read_csv_local("outputs", "tables", "deterministic_event_scale_diagnostics.csv"),
    stochastic = read_csv_local("outputs", "tables", "individual_stochastic_toy_summary.csv")
  )
}

prepare_extended_data_figure_11_data <- function(inputs = load_extended_data_figure_11_inputs()) {
  strategy_labels <- extended_data_figure_11_strategy_labels()
  portfolio_labels <- extended_data_figure_11_portfolio_labels()

  portfolio_bounds <- inputs$portfolio %>%
    pull(iqr_relative_reduction_infant_cases) %>%
    parse_range_bounds()

  portfolio <- inputs$portfolio %>%
    bind_cols(portfolio_bounds) %>%
    mutate(
      portfolio_label = recode(portfolio, !!!portfolio_labels),
      portfolio_label = fct_reorder(portfolio_label, median_relative_reduction_infant_cases),
      resistance_layer = if_else(resistance_guided, "Includes resistance-guided care", "No resistance-guided care"),
      first_label = if_else(countries_ranked_first_infant_cases > 0, paste0(countries_ranked_first_infant_cases, "/10 first"), "")
    )

  contact <- inputs$contact %>%
    filter(strategy %in% c("current", "maternal_immunization")) %>%
    mutate(strategy_label = recode(strategy, !!!strategy_labels))

  maternal <- inputs$maternal %>%
    filter(strategy != "current") %>%
    mutate(strategy_label = recode(strategy, !!!strategy_labels))

  temporal <- inputs$temporal %>%
    mutate(
      scenario_label = recode(
        scenario,
        burnin_10y = "Burn-in 10 y",
        burnin_15y = "Burn-in 15 y",
        burnin_30y = "Burn-in 30 y",
        npi_country_profile = "Country NPI",
        npi_reduction_half = "Half NPI",
        npi_none = "No NPI"
      ),
      temporal_dimension = recode(
        temporal_dimension,
        burn_in = "Burn-in",
        npi_contact_shock = "NPI contact shock"
      ),
      scenario_label = fct_reorder(scenario_label, median_infant_cases_per_100k_5y)
    )

  event_scale <- inputs$event_scale %>%
    mutate(
      scenario_label = recode(scenario, !!!strategy_labels),
      low_event = str_detect(event_scale_flag, "Low infant-event count"),
      scenario_label = factor(
        scenario_label,
        levels = rev(c(
          "Combined stress test", "High transmission blocking", "Infant-exposure strategy",
          "Resistance-guided care", "Close-contact adjunct", "Pregnancy Tdap",
          "Targeted PEP", "Adolescent booster", "Current", "Nominal coverage floor"
        ))
      )
    )

  stochastic <- inputs$stochastic %>%
    mutate(
      country_label = str_replace_all(country, "_", " "),
      scenario_label = recode(
        scenario,
        homogeneous_all_contacts = "Homogeneous",
        setting_clustered = "Setting clustered",
        setting_clustered_high_household = "High household"
      )
    )

  list(
    portfolio = portfolio,
    contact = contact,
    maternal = maternal,
    temporal = temporal,
    event_scale = event_scale,
    stochastic = stochastic
  )
}
