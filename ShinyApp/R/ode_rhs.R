routine_target_distribution_r <- function(params, age) {
  configured <- params$routine_vaccination$target_origin_distribution_by_age
  if (!is.null(configured[[age]])) {
    return(unlist(configured[[age]]))
  }
  default_routine_target_origin_distribution_r(age)
}

routine_delivery_multiplier_at_r <- function(t, params) {
  periods <- params$routine_vaccination$delivery_shock_periods
  if (is.null(periods) || !length(periods)) return(1)
  current <- calendar_date_at_r(params, t)
  multiplier <- 1
  for (period in periods) {
    start <- tryCatch(as.Date(period$start_date), error = function(e) NA)
    end <- tryCatch(as.Date(period$end_date), error = function(e) NA)
    reduction <- as.numeric(period$reduction %||% 0)
    ramp_days <- as.numeric(period$ramp_days %||% 0)
    if (is.na(start) || is.na(end) || end < start || reduction < 0 || reduction > 1) next
    if (current >= start && current <= end) {
      multiplier <- min(multiplier, 1 - reduction)
    } else {
      days_after <- as.numeric(current - end)
      if (days_after > 0 && days_after <= ramp_days) {
        progress <- ifelse(ramp_days <= 0, 1, days_after / ramp_days)
        multiplier <- min(multiplier, (1 - reduction) + reduction * progress)
      }
    }
  }
  clip_probability_r(multiplier)
}

add_routine_vaccination_r <- function(dy, state, params, index, t) {
  config <- params$routine_vaccination
  if (!isTRUE(config$enabled)) return(dy)
  rate <- as.numeric(config$target_relaxation_rate_per_year %||% 0) / 365
  if (rate <= 0) return(dy)
  delivery_multiplier <- routine_delivery_multiplier_at_r(t, params)
  if (delivery_multiplier <= 0) return(dy)
  rate <- rate * delivery_multiplier
  max_flow_fraction <- as.numeric(config$max_daily_flow_fraction %||% 0.01) * delivery_multiplier
  current_population <- pmax(rowSums(state), 0)

  for (age_idx in seq_len(index$n_age)) {
    age <- index$age_groups[[age_idx]]
    target <- routine_target_distribution_r(params, age)
    if (!length(target)) next
    target <- target[names(target) %in% MODEL_VACCINE_ORIGINS & vapply(names(target), origin_is_vaccine_dose_r, logical(1))]
    total_share <- sum(pmax(as.numeric(target), 0))
    if (total_share <= 0) next

    deficits <- numeric(length(target))
    names(deficits) <- names(target)
    for (origin in names(target)) {
      desired <- current_population[[age_idx]] * params$vaccine_coverage[[age_idx]] * max(0, as.numeric(target[[origin]])) / total_share
      current <- state[age_idx, susceptible_name_r(origin)]
      deficits[[origin]] <- max(0, desired - current)
    }
    total_deficit <- sum(deficits)
    if (total_deficit <= 0) next
    s_available <- state[age_idx, "S"]
    total_flow <- min(rate * total_deficit, max_flow_fraction * s_available)
    if (total_flow <= 0) next
    dy[age_idx, "S"] <- dy[age_idx, "S"] - total_flow
    for (origin in names(deficits)) {
      dy[age_idx, susceptible_name_r(origin)] <- dy[age_idx, susceptible_name_r(origin)] +
        total_flow * deficits[[origin]] / total_deficit
    }
  }
  dy
}

add_importation_r <- function(dy, state, params, index) {
  config <- params$importation
  if (!isTRUE(config$enabled)) return(dy)
  rate <- as.numeric(config$rate_per_100k_per_year %||% 0)
  if (rate <= 0) return(dy)
  age_distribution <- vapply(index$age_groups, function(age) as.numeric(config$age_distribution[[age]] %||% (1 / index$n_age)), numeric(1))
  if (sum(age_distribution) <= 0) age_distribution[] <- 1 / index$n_age
  age_distribution <- age_distribution / sum(age_distribution)
  imported <- params$total_population * rate / 100000 / 365 * age_distribution
  resistant_fraction <- clip_probability_r(config$resistant_fraction %||% params$initial$initial_resistance_prevalence %||% 0)

  susceptible_pool <- rep(0, index$n_age)
  for (origin in MODEL_VACCINE_ORIGINS) susceptible_pool <- susceptible_pool + state[, susceptible_name_r(origin)]
  susceptible_pool <- pmax(susceptible_pool, 1e-12)
  for (origin in MODEL_VACCINE_ORIGINS) {
    compartment <- susceptible_name_r(origin)
    share <- state[, compartment] / susceptible_pool
    from_origin <- imported * share
    dy[, compartment] <- dy[, compartment] - pmin(from_origin, state[, compartment])
    dy[, exposed_name_r("R", origin)] <- dy[, exposed_name_r("R", origin)] + from_origin * resistant_fraction
    dy[, exposed_name_r("S", origin)] <- dy[, exposed_name_r("S", origin)] + from_origin * (1 - resistant_fraction)
  }
  dy
}

add_demographic_turnover_r <- function(dy, state, params, index) {
  config <- params$demography
  if (!isTRUE(config$enabled)) return(dy)
  durations <- vapply(index$age_groups, function(age) as.numeric(config$age_bin_durations_years[[age]] %||% 1), numeric(1))
  if (any(durations <= 0)) stop("All demography age-bin durations must be > 0.", call. = FALSE)
  maternal_proxy <- config$maternal_protection_proxy %||% list()
  maternal_exit_groups <- if (isTRUE(maternal_proxy$enabled)) unlist(maternal_proxy$exit_age_groups) else character()

  if (isTRUE(config$fixed_population_profile %||% TRUE)) {
    ref_age <- as.character(config$fixed_population_reference_age_group %||% index$age_groups[[1]])
    ref_idx <- match(ref_age, index$age_groups)
    if (is.na(ref_idx)) ref_idx <- 1
    target_population <- pmax(params$population, 1e-12)
    reference_flow <- target_population[[ref_idx]] / (durations[[ref_idx]] * 365)
    aging_rates <- reference_flow / target_population
  } else {
    aging_rates <- 1 / (durations * 365)
  }

  for (age_idx in seq_len(index$n_age - 1)) {
    flow <- aging_rates[[age_idx]] * state[age_idx, ]
    flow_to_next <- flow
    if (index$age_groups[[age_idx]] %in% maternal_exit_groups) {
      flow_to_next[["S"]] <- flow_to_next[["S"]] + flow_to_next[["M_protected"]]
      flow_to_next[["M_protected"]] <- 0
    }
    dy[age_idx, ] <- dy[age_idx, ] - flow
    dy[age_idx + 1, ] <- dy[age_idx + 1, ] + flow_to_next
  }
  oldest_flow <- aging_rates[[index$n_age]] * state[index$n_age, ]
  dy[index$n_age, ] <- dy[index$n_age, ] - oldest_flow
  total_births <- sum(oldest_flow)
  birth_entry <- config$birth_entry %||% list(S = 1)
  weights <- pmax(as.numeric(unlist(birth_entry)), 0)
  names(weights) <- names(birth_entry)
  if (sum(weights) <= 0) {
    dy[1, "S"] <- dy[1, "S"] + total_births
  } else {
    for (compartment in names(weights)) {
      resolved <- resolve_compartment_name_r(compartment)
      dy[1, resolved] <- dy[1, resolved] + total_births * weights[[compartment]] / sum(weights)
    }
  }
  dy
}

rhs_model_r <- function(t, y, params, index) {
  state <- pmax(state_matrix_r(y, index), 0)
  dy <- matrix(0, nrow = index$n_age, ncol = index$n_compartments,
               dimnames = list(index$age_groups, index$compartments))
  foi <- compute_force_of_infection_r(t, y, params, index)
  lambda <- list(S = foi$lambda_S, R = foi$lambda_R)

  infection_from_origin <- list(S = list(), R = list())
  for (origin in MODEL_VACCINE_ORIGINS) {
    sus_comp <- state[, susceptible_name_r(origin)]
    infection_from_origin$S[[origin]] <- lambda$S * params$origin_susceptibility[[origin]] * sus_comp
    infection_from_origin$R[[origin]] <- lambda$R * params$origin_susceptibility[[origin]] * sus_comp
    dy[, susceptible_name_r(origin)] <- dy[, susceptible_name_r(origin)] -
      infection_from_origin$S[[origin]] - infection_from_origin$R[[origin]]
  }

  dy[, "M_protected"] <- dy[, "M_protected"] - params$rates$waning_maternal * state[, "M_protected"]
  dy[, "S"] <- dy[, "S"] + params$rates$waning_maternal * state[, "M_protected"]
  for (pair in list(c("V_dose1_recent", "V_dose1_waned"), c("V_dose2_recent", "V_dose2_waned"), c("V_recent", "V_waned"))) {
    recent <- pair[[1]]
    waned <- pair[[2]]
    dy[, recent] <- dy[, recent] - params$rates$waning_vaccine * state[, recent]
    dy[, waned] <- dy[, waned] + params$rates$waning_vaccine * state[, recent]
    dy[, waned] <- dy[, waned] - params$rates$waning_vaccine_waned * state[, waned]
    dy[, "S"] <- dy[, "S"] + params$rates$waning_vaccine_waned * state[, waned]
  }

  sigma <- params$rates$latent
  base_gamma_sym <- params$rates$recovery_symptomatic
  base_gamma_asym <- params$rates$recovery_asymptomatic
  gamma_treated_s <- treated_recovery_rate_r(base_gamma_sym, params$treatment, "S")
  gamma_treated_r <- treated_recovery_rate_r(base_gamma_sym, params$treatment, "R")
  tr_sym_by_age <- as.numeric(params$treatment$treatment_rate_symptomatic) * params$diagnosis_probability
  tr_asym_by_age <- as.numeric(params$treatment$treatment_rate_asymptomatic) * params$diagnosis_probability

  recovered <- rep(0, index$n_age)
  for (strain in MODEL_STRAINS) {
    gamma_treated_base <- if (strain == "S") gamma_treated_s else gamma_treated_r
    for (origin in MODEL_VACCINE_ORIGINS) {
      exposed <- exposed_name_r(strain, origin)
      i_sym <- infectious_name_r(strain, "sym", origin)
      i_asym <- infectious_name_r(strain, "asym", origin)
      treated <- treated_name_r(strain, origin)
      progression <- sigma * state[, exposed]
      p_sym <- params$origin_symptomatic_prob[origin, ]
      recovery_multiplier <- params$origin_recovery_mult[[origin]]
      gamma_sym <- base_gamma_sym * recovery_multiplier
      gamma_asym <- base_gamma_asym * recovery_multiplier
      gamma_treated <- gamma_treated_base * recovery_multiplier

      dy[, exposed] <- dy[, exposed] + infection_from_origin[[strain]][[origin]] - progression
      dy[, i_sym] <- dy[, i_sym] + p_sym * progression - tr_sym_by_age * state[, i_sym] - gamma_sym * state[, i_sym]
      dy[, i_asym] <- dy[, i_asym] + (1 - p_sym) * progression - tr_asym_by_age * state[, i_asym] - gamma_asym * state[, i_asym]
      dy[, treated] <- dy[, treated] + tr_sym_by_age * state[, i_sym] + tr_asym_by_age * state[, i_asym] -
        gamma_treated * state[, treated]
      recovered <- recovered + gamma_sym * state[, i_sym] + gamma_asym * state[, i_asym] + gamma_treated * state[, treated]
    }
  }
  dy[, "R_natural"] <- dy[, "R_natural"] + recovered

  boosting_enabled <- isTRUE(params$immunity_model$boosting_enabled %||% TRUE)
  if (boosting_enabled) {
    lambda_total <- lambda$S + lambda$R
    boosting_efficiency <- as.numeric(params$immunity_model$boosting_efficiency %||% 0.7)
    boosting_flow <- boosting_efficiency * lambda_total * state[, "W_natural"]
    breakthrough_susceptibility <- clip_probability_r(params$immunity_model$waned_natural_infection_susceptibility %||% 0.35)
    breakthrough_multiplier <- max(0, 1 - boosting_efficiency) * breakthrough_susceptibility
    breakthrough_s <- breakthrough_multiplier * lambda$S * state[, "W_natural"]
    breakthrough_r <- breakthrough_multiplier * lambda$R * state[, "W_natural"]
    dy[, "R_natural"] <- dy[, "R_natural"] - params$rates$waning_R_to_W * state[, "R_natural"]
    dy[, "W_natural"] <- dy[, "W_natural"] + params$rates$waning_R_to_W * state[, "R_natural"]
    dy[, "W_natural"] <- dy[, "W_natural"] - boosting_flow - params$rates$waning_W_to_S * state[, "W_natural"] -
      breakthrough_s - breakthrough_r
    dy[, "R_natural"] <- dy[, "R_natural"] + boosting_flow
    dy[, "S"] <- dy[, "S"] + params$rates$waning_W_to_S * state[, "W_natural"]
    dy[, exposed_name_r("S", "unvaccinated")] <- dy[, exposed_name_r("S", "unvaccinated")] + breakthrough_s
    dy[, exposed_name_r("R", "unvaccinated")] <- dy[, exposed_name_r("R", "unvaccinated")] + breakthrough_r
  } else {
    dy[, "R_natural"] <- dy[, "R_natural"] - params$rates$waning_natural * state[, "R_natural"]
    dy[, "S"] <- dy[, "S"] + params$rates$waning_natural * state[, "R_natural"]
  }

  dy <- add_routine_vaccination_r(dy, state, params, index, t)
  dy <- add_importation_r(dy, state, params, index)
  dy <- add_demographic_turnover_r(dy, state, params, index)
  state_vector_r(dy)
}
