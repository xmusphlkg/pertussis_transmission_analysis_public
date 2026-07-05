treated_infectiousness_relative_r <- function(treatment, strain) {
  key <- if (strain == "S") "sensitive" else "resistant"
  max(0, 1 - as.numeric(treatment[[key]]$infectiousness_reduction))
}

treated_recovery_rate_r <- function(base_recovery_rate, treatment, strain) {
  key <- if (strain == "S") "sensitive" else "resistant"
  duration_factor <- max(0.05, 1 - as.numeric(treatment[[key]]$infectious_duration_reduction))
  as.numeric(base_recovery_rate) / duration_factor
}

npi_contact_reduction_at_r <- function(t, params) {
  periods <- params$transmission$npi_contact_reduction_periods
  if (is.null(periods) || !length(periods)) return(1)
  current <- calendar_date_at_r(params, t)
  best <- 1
  for (period in periods) {
    start <- tryCatch(as.Date(period$start_date), error = function(e) NA)
    end <- tryCatch(as.Date(period$end_date), error = function(e) NA)
    reduction <- as.numeric(period$reduction %||% 0)
    ramp_days <- as.numeric(period$ramp_days %||% params$transmission$npi_ramp_days %||% 90)
    if (is.na(start) || is.na(end) || end < start || reduction < 0 || reduction > 1) next
    if (current >= start && current <= end) {
      best <- min(best, 1 - reduction)
    } else {
      days_after <- as.numeric(current - end)
      if (days_after > 0 && days_after <= ramp_days) {
        progress <- ifelse(ramp_days <= 0, 1, days_after / ramp_days)
        best <- min(best, (1 - reduction) + reduction * progress)
      }
    }
  }
  clip_probability_r(best)
}

seasonal_multiplier_r <- function(t, params) {
  amplitude <- as.numeric(params$transmission$seasonal_amplitude %||% 0)
  phase <- as.numeric(params$transmission$seasonal_phase %||% 0)
  day <- calendar_day_of_year_at_r(params, t)
  annual <- 1 + amplitude * cos(2 * pi * (day - phase) / 365)

  multi_amp <- as.numeric(params$transmission$multi_year_amplitude %||% 0)
  multi_period_days <- 365 * as.numeric(params$transmission$multi_year_period_years %||% 4)
  multi_phase <- as.numeric(params$transmission$multi_year_phase %||% 0)
  multi_year <- if (multi_period_days <= 0) {
    1
  } else {
    1 + multi_amp * cos(2 * pi * (as.numeric(t) - multi_phase) / multi_period_days)
  }
  max(0, annual * multi_year * npi_contact_reduction_at_r(t, params))
}

compute_force_of_infection_r <- function(t, y, params, index, apply_pep = TRUE) {
  state <- pmax(state_matrix_r(y, index), 0)
  population <- pmax(rowSums(state), 1)
  rel_asym <- as.numeric(params$transmission$relative_infectiousness_asymptomatic)
  rel_treated_s <- treated_infectiousness_relative_r(params$treatment, "S")
  rel_treated_r <- treated_infectiousness_relative_r(params$treatment, "R")

  pressure_by_strain <- list()
  for (strain in MODEL_STRAINS) {
    treated_relative <- if (strain == "S") rel_treated_s else rel_treated_r
    pressure <- rep(0, index$n_age)
    for (origin in MODEL_VACCINE_ORIGINS) {
      origin_inf <- params$origin_infectiousness[[origin]]
      pressure <- pressure + origin_inf * (
        state[, infectious_name_r(strain, "sym", origin)] +
          rel_asym * state[, infectious_name_r(strain, "asym", origin)] +
          treated_relative * state[, treated_name_r(strain, origin)]
      )
    }
    pressure_by_strain[[strain]] <- pressure / population
  }

  beta_s <- as.numeric(params$transmission$beta_S) * seasonal_multiplier_r(t, params)
  beta_r <- beta_s * as.numeric(params$transmission$fitness_R %||% 1)
  lambda_s_base <- as.numeric(beta_s * (params$contact_matrix %*% pressure_by_strain$S))
  lambda_r_base <- as.numeric(beta_r * (params$contact_matrix %*% pressure_by_strain$R))

  lambda_s <- lambda_s_base
  lambda_r <- lambda_r_base
  pep_coverage_by_age <- rep(0, index$n_age)
  if (isTRUE(apply_pep)) {
    symptomatic <- rep(0, index$n_age)
    for (strain in MODEL_STRAINS) {
      for (origin in MODEL_VACCINE_ORIGINS) {
        symptomatic <- symptomatic + state[, infectious_name_r(strain, "sym", origin)]
      }
    }
    detected_prev <- symptomatic * params$pep_detection_rate / population
    contact_weighted <- as.numeric(params$contact_matrix %*% detected_prev)
    activation <- contact_weighted / (contact_weighted + as.numeric(params$PEP$activation_prevalence %||% 1e-5))
    pep_coverage_by_age <- as.numeric(params$PEP$coverage_household_contacts %||% 0) * activation
    lambda_s <- lambda_s * (1 - pep_coverage_by_age * as.numeric(params$PEP$effectiveness_sensitive %||% 0))
    lambda_r <- lambda_r * (1 - pep_coverage_by_age * as.numeric(params$PEP$effectiveness_resistant %||% 0))
  }

  list(
    lambda_S = pmax(lambda_s, 0),
    lambda_R = pmax(lambda_r, 0),
    lambda_S_base = pmax(lambda_s_base, 0),
    lambda_R_base = pmax(lambda_r_base, 0),
    pep_coverage = ifelse(sum(population) > 0, stats::weighted.mean(pep_coverage_by_age, population), 0),
    pep_coverage_by_age = pep_coverage_by_age
  )
}
