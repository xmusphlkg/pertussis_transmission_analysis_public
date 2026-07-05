resolve_calendar_horizon_r <- function(config) {
  sim <- config$simulation
  cal <- config$calendar
  start_date <- as.Date(cal$analysis_start_date %||% "2025-01-01")
  end_date <- as.Date(cal$analysis_end_date %||% "2035-12-31")
  if (!is.finite(as.numeric(end_date - start_date)) || end_date <= start_date) {
    stop("calendar.analysis_end_date must be after analysis_start_date.", call. = FALSE)
  }
  sim$start_time <- as.numeric(sim$start_time %||% 0)
  sim$end_time <- as.numeric(end_date - start_date)
  config$simulation <- sim
  config
}

apply_country_profile_r <- function(config, country, profile) {
  out <- config
  for (i in seq_along(out$age_groups)) {
    label <- out$age_groups[[i]]$label
    if (!is.null(profile$population[[label]])) {
      out$age_groups[[i]]$population <- as.numeric(profile$population[[label]])
    }
    if (!is.null(profile$reporting_rate[[label]])) {
      out$age_groups[[i]]$reporting_rate <- as.numeric(profile$reporting_rate[[label]])
      pep <- profile$pep_detection_rate %||% profile$reporting_rate
      out$age_groups[[i]]$pep_detection_rate <- as.numeric(pep[[label]] %||% out$age_groups[[i]]$reporting_rate)
    }
    if (!is.null(profile$vaccine_coverage[[label]])) {
      out$age_groups[[i]]$vaccine_coverage <- as.numeric(profile$vaccine_coverage[[label]])
    }
  }
  if (!is.null(profile$contact_matrix)) {
    out$contact_matrix$rows <- profile$contact_matrix
  }
  if (!is.null(profile$birth_entry)) {
    out$demography$birth_entry <- profile$birth_entry
  }
  if (!is.null(profile$transmission_overrides)) {
    out$transmission <- deep_update_r(out$transmission, profile$transmission_overrides)
  }
  if (!is.null(profile$vaccine_schedule)) {
    out$metadata <- out$metadata %||% list()
    for (key in c(
      "routine_age_pattern",
      "routine_first_shot_months",
      "routine_last_shot_months",
      "routine_dose_count",
      "routine_scheduler_code"
    )) {
      if (!is.null(profile$vaccine_schedule[[key]])) out$metadata[[key]] <- profile$vaccine_schedule[[key]]
    }
  }
  out$country <- country
  out
}

apply_vaccine_scenario_r <- function(config, vaccine) {
  out <- config
  out$vaccine <- vaccine[grepl("^VE_", names(vaccine))]
  out
}

apply_resistance_scenario_r <- function(config, resistance) {
  out <- config
  target <- as.numeric(resistance$target_prevalence_at_analysis_start %||% resistance$initial_resistance_prevalence %||% 0)
  out$initial_conditions$initial_resistance_prevalence <- as.numeric(resistance$initial_resistance_prevalence %||% target)
  out$resistance <- out$resistance %||% list()
  out$resistance$target_prevalence_at_analysis_start <- target
  out$resistance$importation_fraction <- as.numeric(resistance$importation_fraction %||% target)
  out$resistance$rebalance_after_burn_in <- isTRUE(resistance$rebalance_after_burn_in %||% TRUE)
  out$resistance$prevalence_anchor_rate_per_year <- as.numeric(
    resistance$prevalence_anchor_rate_per_year %||% out$resistance$prevalence_anchor_rate_per_year %||% 0
  )
  out$resistance$anchor_during_dynamics <- isTRUE(resistance$anchor_during_dynamics %||% FALSE)
  out$resistance$use_country_resistance_timeline <- isTRUE(resistance$use_country_resistance_timeline %||% FALSE)
  out$importation <- out$importation %||% list()
  out$importation$resistant_fraction <- out$resistance$importation_fraction
  out$transmission$fitness_R <- as.numeric(resistance$fitness_R %||% out$transmission$fitness_R %||% 1)
  out
}

country_resistance_estimate_r <- function(country, anchor_year = 2025) {
  path <- app_path_r("data", "country_resistance_timeline.csv")
  if (!file.exists(path)) return(NULL)
  timeline <- read_csv_r(path)
  rows <- timeline[timeline$country == country | timeline$iso3 == country, , drop = FALSE]
  rows <- rows[!is.na(rows$year) & rows$year <= anchor_year, , drop = FALSE]
  if (!nrow(rows)) return(NULL)
  rows <- rows[order(rows$year, decreasing = TRUE), , drop = FALSE]
  list(
    resistant_fraction = as.numeric(rows$resistant_fraction[[1]]),
    evidence_year = as.integer(rows$year[[1]]),
    evidence_type = as.character(rows$evidence_type[[1]])
  )
}

apply_country_resistance_timeline_r <- function(config, country, configs) {
  out <- config
  anchor_year <- as.integer(configs$data_sources$resistance_anchor_year %||% 2025)
  estimate <- country_resistance_estimate_r(country, anchor_year)
  if (is.null(estimate)) return(out)
  value <- clip_probability_r(estimate$resistant_fraction)
  out$resistance$target_prevalence_at_analysis_start <- value
  out$resistance$importation_fraction <- value
  out$importation$resistant_fraction <- value
  out$initial_conditions$initial_resistance_prevalence <- value
  out$metadata <- out$metadata %||% list()
  out$metadata$resistance_anchor_year <- estimate$evidence_year
  out$metadata$resistance_anchor_type <- estimate$evidence_type
  out
}

make_config_r <- function(country_profile = NULL,
                          vaccine_scenario = NULL,
                          resistance_scenario = NULL,
                          vaccine_overrides = NULL,
                          resistance_overrides = NULL,
                          config_overrides = NULL,
                          load_calibration = TRUE,
                          configs = load_app_configs_r()) {
  base <- configs$baseline
  out <- base
  country <- country_profile %||% base$baseline_country_profile %||% "China"
  if (!is.null(configs$countries[[country]])) {
    out <- apply_country_profile_r(out, country, configs$countries[[country]])
  }
  if (isTRUE(load_calibration)) {
    out <- deep_update_r(out, load_calibration_overlay_r(country))
    out$metadata <- out$metadata %||% list()
    out$metadata$calibration_loaded <- length(load_calibration_overlay_r(country)) > 0
  }

  vaccine_name <- vaccine_scenario %||% base$baseline_vaccine_scenario %||% "symptom_protective"
  resistance_name <- resistance_scenario %||% base$baseline_resistance_scenario %||% "country_timeline"
  vaccine <- configs$vacccines[[vaccine_name]]
  if (is.null(vaccine)) vaccine <- configs$vaccines[[vaccine_name]]
  if (is.null(vaccine)) stop("Unknown vaccine scenario: ", vaccine_name, call. = FALSE)
  if (!is.null(vaccine_overrides)) vaccine <- deep_update_r(vaccine, vaccine_overrides)
  out <- apply_vaccine_scenario_r(out, vaccine)

  resistance <- configs$resistance[[resistance_name]]
  if (is.null(resistance)) stop("Unknown resistance scenario: ", resistance_name, call. = FALSE)
  if (!is.null(resistance_overrides)) resistance <- deep_update_r(resistance, resistance_overrides)
  out <- apply_resistance_scenario_r(out, resistance)
  if (isTRUE(out$resistance$use_country_resistance_timeline) && is.null(resistance_overrides)) {
    out <- apply_country_resistance_timeline_r(out, country, configs)
  }

  if (!is.null(config_overrides)) out <- deep_update_r(out, config_overrides)
  if (sum(as.numeric(unlist(out$vaccine[grepl("^VE_", names(out$vaccine))]))) == 0) {
    for (i in seq_along(out$age_groups)) out$age_groups[[i]]$vaccine_coverage <- 0
    out$demography$birth_entry <- list(S = 1, V = 0)
  }
  resolve_calendar_horizon_r(out)
}

contact_matrix_from_config_r <- function(config, age_groups = MODEL_AGE_GROUPS) {
  rows <- config$contact_matrix$rows
  mat <- matrix(as.numeric(unlist(rows)), nrow = length(age_groups), byrow = TRUE)
  rownames(mat) <- age_groups
  colnames(mat) <- age_groups
  validate_contact_matrix_r(mat)
}

prepare_params_r <- function(config,
                             analysis = "interactive",
                             scenario = "current",
                             vaccine_scenario = "",
                             resistance_scenario = "",
                             intervention = "",
                             metadata = list()) {
  age_records <- config$age_groups
  age_groups <- vapply(age_records, `[[`, character(1), "label")
  population <- vapply(age_records, function(x) as.numeric(x$population), numeric(1))
  if (any(!is.finite(population)) || any(population <= 0)) {
    stop("Age-group populations must be finite and > 0.", call. = FALSE)
  }

  vaccine <- config$vaccine %||% list(VE_sus = 0, VE_sym = 0, VE_inf = 0, VE_dur = 0)
  for (key in c("VE_sus", "VE_sym", "VE_inf", "VE_dur")) vaccine[[key]] <- as.numeric(vaccine[[key]] %||% 0)
  assert_probability_r(unlist(vaccine), "Vaccine efficacy values")

  vaccine_coverage <- vapply(age_records, function(x) as.numeric(x$vaccine_coverage %||% 0), numeric(1))
  if (sum(unlist(vaccine)) == 0) vaccine_coverage[] <- 0
  assert_probability_r(vaccine_coverage, "Age-group vaccine coverage")

  reporting_multiplier <- as.numeric(config$reporting_multiplier %||% 1)
  reporting_base <- vapply(age_records, function(x) as.numeric(x$reporting_rate %||% 0), numeric(1))
  reporting_rate <- clip_probability_r(reporting_base * reporting_multiplier)
  pep_detection_rate <- vapply(age_records, function(x) as.numeric(x$pep_detection_rate %||% x$reporting_rate %||% 0), numeric(1))
  assert_probability_r(reporting_rate, "Age-group reporting rates")
  assert_probability_r(pep_detection_rate, "Age-group PEP detection rates")

  diagnosis_cfg <- config$diagnosis_probability
  if (is.null(diagnosis_cfg)) {
    diagnosis_probability <- reporting_base
  } else {
    diagnosis_probability <- vapply(age_groups, function(age) as.numeric(diagnosis_cfg[[age]] %||% reporting_base[[which(age_groups == age)]]), numeric(1))
  }
  diagnosis_probability <- clip_probability_r(diagnosis_probability)

  natural <- config$natural_history
  immunity <- config$immunity_model %||% list()
  waned_vaccine_duration <- as.numeric(
    immunity$waned_vaccine_duration %||% natural$waned_vaccine_duration %||% natural$vaccine_protection_duration
  )
  rates <- list(
    latent = 1 / as.numeric(natural$latent_duration),
    recovery_symptomatic = 1 / as.numeric(natural$infectious_duration_symptomatic),
    recovery_asymptomatic = 1 / as.numeric(natural$infectious_duration_asymptomatic),
    waning_natural = 1 / as.numeric(natural$recovered_immunity_duration),
    waning_vaccine = 1 / as.numeric(natural$vaccine_protection_duration),
    waning_vaccine_waned = ifelse(waned_vaccine_duration > 0, 1 / waned_vaccine_duration, 0),
    waning_maternal = 1 / as.numeric(natural$maternal_protection_duration %||% 90),
    waning_R_to_W = 1 / as.numeric(natural$R_to_W_duration %||% natural$recovered_immunity_duration),
    waning_W_to_S = 1 / as.numeric(natural$W_to_S_duration %||% (as.numeric(natural$recovered_immunity_duration) * 2))
  )
  if (!is.null(config$rates)) rates <- deep_update_r(rates, config$rates)

  symptom_probability <- vapply(age_records, function(x) as.numeric(x$symptom_probability %||% 0.4), numeric(1))
  assert_probability_r(symptom_probability, "Age-group symptom probabilities")

  waned_rel <- as.numeric(immunity$waned_relative_effect %||% 0.35)
  maternal_rel <- as.numeric(immunity$maternal_relative_effect %||% 0.75)
  dose1_rel <- as.numeric(immunity$dose1_relative_effect %||% 0.45)
  dose2_rel <- as.numeric(immunity$dose2_relative_effect %||% 0.75)
  rel_effects <- vapply(
    MODEL_VACCINE_ORIGINS,
    origin_relative_effect_r,
    numeric(1),
    waned_relative_effect = waned_rel,
    maternal_relative_effect = maternal_rel,
    dose1_relative_effect = dose1_rel,
    dose2_relative_effect = dose2_rel
  )

  maternal_ve_sus <- as.numeric(immunity$maternal_VE_sus %||% vaccine$VE_sus)
  maternal_ve_sym <- as.numeric(immunity$maternal_VE_sym %||% vaccine$VE_sym)
  maternal_ve_inf <- as.numeric(immunity$maternal_VE_inf %||% vaccine$VE_inf)
  maternal_ve_dur <- as.numeric(immunity$maternal_VE_dur %||% vaccine$VE_dur)

  origin_susceptibility <- numeric(length(MODEL_VACCINE_ORIGINS))
  origin_infectiousness <- numeric(length(MODEL_VACCINE_ORIGINS))
  origin_recovery_mult <- numeric(length(MODEL_VACCINE_ORIGINS))
  origin_symptomatic_prob <- matrix(0, nrow = length(MODEL_VACCINE_ORIGINS), ncol = length(age_groups),
                                    dimnames = list(MODEL_VACCINE_ORIGINS, age_groups))
  for (i in seq_along(MODEL_VACCINE_ORIGINS)) {
    origin <- MODEL_VACCINE_ORIGINS[[i]]
    eff <- rel_effects[[i]]
    ve_sus_origin <- if (origin == "maternal") maternal_ve_sus else vaccine$VE_sus
    ve_sym_origin <- if (origin == "maternal") maternal_ve_sym else vaccine$VE_sym
    ve_inf_origin <- if (origin == "maternal") maternal_ve_inf else vaccine$VE_inf
    ve_dur_origin <- if (origin == "maternal") maternal_ve_dur else vaccine$VE_dur
    origin_susceptibility[[i]] <- vaccine_susceptibility_r(ve_sus_origin, eff)
    origin_infectiousness[[i]] <- clip_probability_r(1 - ve_inf_origin * eff)
    origin_symptomatic_prob[i, ] <- clip_probability_r(symptom_probability * (1 - ve_sym_origin * eff))
    origin_recovery_mult[[i]] <- 1 / max(0.05, 1 - ve_dur_origin * eff)
  }
  names(origin_susceptibility) <- MODEL_VACCINE_ORIGINS
  names(origin_infectiousness) <- MODEL_VACCINE_ORIGINS
  names(origin_recovery_mult) <- MODEL_VACCINE_ORIGINS

  list(
    raw = config,
    analysis = analysis,
    scenario = scenario,
    vaccine_scenario = vaccine_scenario,
    resistance_scenario = resistance_scenario,
    intervention = intervention,
    age_groups = age_groups,
    population = population,
    vaccine_coverage = vaccine_coverage,
    symptom_probability = symptom_probability,
    reporting_rate = reporting_rate,
    diagnosis_probability = diagnosis_probability,
    pep_detection_rate = pep_detection_rate,
    contact_matrix = contact_matrix_from_config_r(config, age_groups),
    vaccine = vaccine,
    rates = rates,
    transmission = config$transmission,
    treatment = config$treatment,
    PEP = config$PEP,
    initial = config$initial_conditions,
    immunity_model = immunity,
    observation_model = config$observation_model %||% list(),
    resistance = config$resistance %||% list(),
    demography = config$demography %||% list(enabled = FALSE),
    routine_vaccination = config$routine_vaccination %||% list(enabled = FALSE),
    importation = config$importation %||% list(enabled = FALSE),
    calendar = config$calendar %||% list(),
    calendar_start_date = as.Date((config$calendar %||% list())$analysis_start_date %||% "2025-01-01"),
    reporting_multiplier = reporting_multiplier,
    metadata = c(config$metadata %||% list(), metadata),
    origin_relative_effects = rel_effects,
    origin_susceptibility = origin_susceptibility,
    origin_infectiousness = origin_infectiousness,
    origin_symptomatic_prob = origin_symptomatic_prob,
    origin_recovery_mult = origin_recovery_mult,
    total_population = sum(population)
  )
}

calendar_date_at_r <- function(params, t) {
  start_time <- as.numeric(params$raw$simulation$start_time %||% 0)
  params$calendar_start_date + round(as.numeric(t) - start_time)
}

calendar_day_of_year_at_r <- function(params, t) {
  day <- as.POSIXlt(calendar_date_at_r(params, t))$yday + 1
  min(as.numeric(day), 365)
}
