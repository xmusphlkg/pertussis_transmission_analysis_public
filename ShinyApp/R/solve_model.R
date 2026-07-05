initial_origin_distribution_r <- function(params, age) {
  configured <- params$immunity_model$initial_origin_distribution_by_age
  if (!is.null(configured[[age]])) {
    return(unlist(configured[[age]]))
  }
  if (identical(params$immunity_model$mode, "recent_waned_proxy")) {
    recent_fraction <- as.numeric((params$immunity_model$initial_recent_fraction_by_age %||% list())[[age]] %||%
      params$immunity_model$initial_recent_fraction %||% 0.5)
    recent_fraction <- clip_probability_r(recent_fraction)
    if (age == "infant_0_2m") return(c(maternal = 1))
    if (age == "infant_3_11m") return(c(dose1_recent = 0.25, dose2_recent = 0.35, recent = 0.40))
    return(c(recent = recent_fraction, waned = 1 - recent_fraction))
  }
  default_initial_origin_distribution_r(age)
}

target_resistance_prevalence_r <- function(params) {
  clip_probability_r(params$resistance$target_prevalence_at_analysis_start %||%
    params$initial$initial_resistance_prevalence %||% 0)
}

initial_state_r <- function(params, index) {
  state <- matrix(0, nrow = index$n_age, ncol = index$n_compartments,
                  dimnames = list(index$age_groups, index$compartments))
  total_seed_exposed <- params$total_population * as.numeric(params$initial$initial_exposed_per_100k %||% 0) / 100000
  total_seed_infectious <- params$total_population * as.numeric(params$initial$initial_infectious_per_100k %||% 0) / 100000
  resistant_fraction <- clip_probability_r(params$initial$initial_resistance_prevalence %||% 0)
  seed_distribution <- params$initial$seed_age_distribution %||% list()

  for (age_idx in seq_len(index$n_age)) {
    age <- index$age_groups[[age_idx]]
    coverage <- params$vaccine_coverage[[age_idx]]
    origin_distribution <- initial_origin_distribution_r(params, age)
    origin_total <- max(sum(pmax(as.numeric(origin_distribution), 0)), 1e-12)
    for (origin in names(origin_distribution)) {
      if (!origin %in% MODEL_VACCINE_ORIGINS || origin == "unvaccinated") next
      state[age_idx, susceptible_name_r(origin)] <- state[age_idx, susceptible_name_r(origin)] +
        params$population[[age_idx]] * coverage * max(0, as.numeric(origin_distribution[[origin]])) / origin_total
    }
    state[age_idx, "S"] <- params$population[[age_idx]] * (1 - coverage)

    share <- as.numeric(seed_distribution[[age]] %||% (1 / index$n_age))
    exposed <- total_seed_exposed * share
    infectious <- total_seed_infectious * share
    seeded <- exposed + infectious
    source_pools <- vapply(MODEL_VACCINE_ORIGINS, function(origin) state[age_idx, susceptible_name_r(origin)], numeric(1))
    source_total <- max(sum(source_pools), 1e-12)

    for (origin in MODEL_VACCINE_ORIGINS) {
      origin_share <- source_pools[[origin]] / source_total
      origin_exposed <- exposed * origin_share
      origin_infectious <- infectious * origin_share
      p_sym <- params$origin_symptomatic_prob[origin, age_idx]
      state[age_idx, exposed_name_r("R", origin)] <- origin_exposed * resistant_fraction
      state[age_idx, exposed_name_r("S", origin)] <- origin_exposed * (1 - resistant_fraction)
      state[age_idx, infectious_name_r("R", "sym", origin)] <- origin_infectious * resistant_fraction * p_sym
      state[age_idx, infectious_name_r("R", "asym", origin)] <- origin_infectious * resistant_fraction * (1 - p_sym)
      state[age_idx, infectious_name_r("S", "sym", origin)] <- origin_infectious * (1 - resistant_fraction) * p_sym
      state[age_idx, infectious_name_r("S", "asym", origin)] <- origin_infectious * (1 - resistant_fraction) * (1 - p_sym)
    }

    for (origin in MODEL_VACCINE_ORIGINS) {
      compartment <- susceptible_name_r(origin)
      state[age_idx, compartment] <- max(0, state[age_idx, compartment] - seeded * source_pools[[origin]] / source_total)
    }
  }
  state_vector_r(state)
}

rebalance_resistant_prevalence_r <- function(y, params, index, target = NULL) {
  target_fraction <- if (is.null(target)) target_resistance_prevalence_r(params) else clip_probability_r(target)
  state <- pmax(state_matrix_r(y, index), 0)
  for (origin in MODEL_VACCINE_ORIGINS) {
    pairs <- list(
      c(exposed_name_r("S", origin), exposed_name_r("R", origin)),
      c(infectious_name_r("S", "sym", origin), infectious_name_r("R", "sym", origin)),
      c(infectious_name_r("S", "asym", origin), infectious_name_r("R", "asym", origin)),
      c(treated_name_r("S", origin), treated_name_r("R", origin))
    )
    for (pair in pairs) {
      total <- state[, pair[[1]]] + state[, pair[[2]]]
      state[, pair[[2]]] <- total * target_fraction
      state[, pair[[1]]] <- total * (1 - target_fraction)
    }
  }
  state_vector_r(state)
}

solve_model_r <- function(params, index = make_state_index_r(params$age_groups)) {
  require_namespace_r("deSolve")
  sim <- params$raw$simulation
  output_time_step <- as.numeric(sim$output_time_step %||% sim$time_step %||% 30)
  start_time <- as.numeric(sim$start_time %||% 0)
  end_time <- as.numeric(sim$end_time)
  times <- seq(start_time, end_time, by = output_time_step)
  if (tail(times, 1) < end_time) times <- c(times, end_time)
  y0 <- initial_state_r(params, index)

  burn_in_years <- as.numeric(sim$burn_in_years %||% 0)
  method <- tolower(as.character(sim$solver_method %||% "lsoda"))
  if (method %in% c("rk4", "fixed_rk4")) {
    dt_analysis <- as.numeric(sim$rk4_dt_analysis %||% 14)
    dt_burnin <- as.numeric(sim$rk4_dt_burnin %||% max(dt_analysis, 30))
    if (burn_in_years > 0) {
      burn_start <- start_time - burn_in_years * 365
      y0 <- rk4_integrate_r(
        y0,
        t_start = burn_start,
        t_end = start_time,
        params = params,
        index = index,
        dt = dt_burnin,
        output_times = start_time
      )
      y0 <- pmax(as.numeric(y0[nrow(y0), -1]), 0)
      if (isTRUE(params$resistance$rebalance_after_burn_in %||% TRUE)) {
        y0 <- rebalance_resistant_prevalence_r(y0, params, index)
      }
    }
    out <- rk4_integrate_r(
      y0,
      t_start = start_time,
      t_end = end_time,
      params = params,
      index = index,
      dt = dt_analysis,
      output_times = times
    )
    attr(out, "params") <- params
    attr(out, "index") <- index
    return(out)
  }
  if (method %in% c("rk45", "bdf")) method <- "lsoda"
  rtol <- as.numeric(sim$rtol %||% 1e-5)
  atol <- as.numeric(sim$atol %||% 1e-7)

  if (burn_in_years > 0) {
    burn_start <- start_time - burn_in_years * 365
    burn_out <- deSolve::ode(
      y = y0,
      times = c(burn_start, start_time),
      func = function(t, y, parms) list(rhs_model_r(t, y, parms$params, parms$index)),
      parms = list(params = params, index = index),
      method = method,
      rtol = rtol,
      atol = atol
    )
    y0 <- pmax(as.numeric(burn_out[nrow(burn_out), -1]), 0)
    if (isTRUE(params$resistance$rebalance_after_burn_in %||% TRUE)) {
      y0 <- rebalance_resistant_prevalence_r(y0, params, index)
    }
  }

  out <- deSolve::ode(
    y = y0,
    times = times,
    func = function(t, y, parms) list(rhs_model_r(t, y, parms$params, parms$index)),
    parms = list(params = params, index = index),
    method = method,
    rtol = rtol,
    atol = atol
  )
  attr(out, "params") <- params
  attr(out, "index") <- index
  out
}

rk4_integrate_r <- function(y0, t_start, t_end, params, index, dt, output_times) {
  if (!is.finite(dt) || dt <= 0) {
    stop("RK4 step size must be finite and > 0.", call. = FALSE)
  }
  output_times <- sort(unique(as.numeric(output_times)))
  output_times <- output_times[output_times >= t_start - 1e-9 & output_times <= t_end + 1e-9]
  if (!length(output_times) || abs(output_times[[1]] - t_start) > 1e-9) {
    output_times <- sort(unique(c(t_start, output_times)))
  }
  y <- as.numeric(y0)
  t <- as.numeric(t_start)
  rows <- matrix(NA_real_, nrow = length(output_times), ncol = length(y) + 1)
  colnames(rows) <- c("time", paste0("V", seq_along(y)))
  out_i <- 1

  record_if_due <- function() {
    while (out_i <= length(output_times) && abs(t - output_times[[out_i]]) <= 1e-7) {
      rows[out_i, ] <<- c(t, y)
      out_i <<- out_i + 1
    }
  }
  record_if_due()
  while (t < t_end - 1e-9) {
    next_output <- if (out_i <= length(output_times)) output_times[[out_i]] else t_end
    h <- min(dt, t_end - t, next_output - t)
    if (h <= 1e-9) {
      t <- next_output
      record_if_due()
      next
    }
    k1 <- rhs_model_r(t, y, params, index)
    k2 <- rhs_model_r(t + 0.5 * h, pmax(y + 0.5 * h * k1, 0), params, index)
    k3 <- rhs_model_r(t + 0.5 * h, pmax(y + 0.5 * h * k2, 0), params, index)
    k4 <- rhs_model_r(t + h, pmax(y + h * k3, 0), params, index)
    y <- pmax(y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0)
    t <- t + h
    record_if_due()
  }
  rows <- rows[seq_len(out_i - 1), , drop = FALSE]
  class(rows) <- c("deSolve", "matrix")
  rows
}

run_prepared_config_r <- function(config,
                                  analysis = "interactive",
                                  scenario = "current",
                                  vaccine_scenario = "",
                                  resistance_scenario = "",
                                  intervention = "",
                                  metadata = list()) {
  params <- prepare_params_r(
    config,
    analysis = analysis,
    scenario = scenario,
    vaccine_scenario = vaccine_scenario,
    resistance_scenario = resistance_scenario,
    intervention = intervention,
    metadata = metadata
  )
  index <- make_state_index_r(params$age_groups)
  solution <- solve_model_r(params, index)
  timeseries <- compute_timeseries_r(solution, params, index)
  summary <- summarize_timeseries_r(timeseries)
  for (nm in names(params$metadata)) {
    value <- params$metadata[[nm]]
    if (length(value) == 1 && (is.character(value) || is.numeric(value) || is.logical(value))) {
      timeseries[[nm]] <- value
      summary[[nm]] <- value
    }
  }
  list(timeseries = timeseries, summary = summary)
}
