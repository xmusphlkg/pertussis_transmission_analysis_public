STRATEGY_LABELS_R <- c(
  current = "Current practice",
  higher_child_coverage = "Higher child coverage",
  timeliness_only = "Routine timeliness",
  adolescent_booster = "Adolescent booster",
  pregnancy_tdap_scaleup = "Pregnancy Tdap scale-up",
  cocooning_adjunct = "Close-contact adult adjunct",
  maternal_immunization = "Infant-exposure reduction",
  targeted_pep_high_risk = "Targeted high-risk PEP",
  resistance_guided_treatment = "Resistance-guided management",
  transmission_blocking_vaccine = "Transmission-blocking vaccine target",
  next_generation_vaccine = "High-transmission-blocking vaccine target",
  combined_strategy = "Combined future stress-test profile"
)

STRATEGY_INTENSITY_R <- c(
  current = 0,
  higher_child_coverage = 1,
  timeliness_only = 1,
  adolescent_booster = 1,
  pregnancy_tdap_scaleup = 1,
  targeted_pep_high_risk = 1,
  cocooning_adjunct = 2,
  resistance_guided_treatment = 2,
  maternal_immunization = 3,
  transmission_blocking_vaccine = 4,
  next_generation_vaccine = 4,
  combined_strategy = 5
)

DEFAULT_STRATEGIES_R <- names(STRATEGY_LABELS_R)

set_age_group_value_r <- function(config, age_group, field, value, mode = c("set", "min")) {
  mode <- match.arg(mode)
  out <- config
  for (i in seq_along(out$age_groups)) {
    if (identical(out$age_groups[[i]]$label, age_group)) {
      current <- as.numeric(out$age_groups[[i]][[field]] %||% 0)
      out$age_groups[[i]][[field]] <- if (mode == "min") max(current, as.numeric(value)) else as.numeric(value)
      break
    }
  }
  out
}

apply_coverage_updates_r <- function(config, updates, mode = c("set", "min")) {
  mode <- match.arg(mode)
  out <- config
  if (is.null(updates)) return(out)
  for (age in names(updates)) {
    out <- set_age_group_value_r(out, age, "vaccine_coverage", clip_probability_r(updates[[age]]), mode)
  }
  out
}

contact_rows_from_matrix_r <- function(mat) {
  lapply(seq_len(nrow(mat)), function(i) as.numeric(mat[i, ]))
}

apply_intervention_definition_r <- function(config, intervention) {
  out <- config
  coverage_updates <- intervention$coverage_updates %||% list()
  coverage_min_updates <- intervention$coverage_min_updates %||% list()
  out <- apply_coverage_updates_r(out, coverage_updates, "set")
  out <- apply_coverage_updates_r(out, coverage_min_updates, "min")

  if (!is.null(intervention$vaccine_overrides)) {
    overrides <- intervention$vaccine_overrides
    maternal_keys <- names(overrides)[grepl("^maternal_", names(overrides))]
    global_keys <- setdiff(names(overrides), maternal_keys)
    for (key in global_keys) out$vaccine[[key]] <- overrides[[key]]
    out$immunity_model <- out$immunity_model %||% list()
    for (key in maternal_keys) out$immunity_model[[key]] <- overrides[[key]]
  }
  if (!is.null(intervention$treatment_updates)) {
    out$treatment <- deep_update_r(out$treatment, intervention$treatment_updates)
  }
  if (!is.null(intervention$pep_updates)) {
    out$PEP <- deep_update_r(out$PEP, intervention$pep_updates)
  }
  if (!is.null(intervention$natural_history_overrides)) {
    out$natural_history <- deep_update_r(out$natural_history, intervention$natural_history_overrides)
  }

  maternal_cov <- NULL
  if (!is.null(coverage_updates$infant_0_2m)) {
    maternal_cov <- clip_probability_r(coverage_updates$infant_0_2m)
  } else if (!is.null(coverage_min_updates$infant_0_2m)) {
    existing <- clip_probability_r((out$demography$birth_entry %||% list())$V %||% 0)
    maternal_cov <- max(existing, clip_probability_r(coverage_min_updates$infant_0_2m))
  }
  if (!is.null(maternal_cov)) {
    out$demography$birth_entry <- list(S = 1 - maternal_cov, V = maternal_cov)
  }

  cmr <- intervention$contact_matrix_reduction
  if (!is.null(cmr)) {
    age_labels <- vapply(out$age_groups, `[[`, character(1), "label")
    mat <- contact_matrix_from_config_r(out, age_labels)
    source_age <- as.character(cmr$source_age)
    target_ages <- unlist(cmr$target_ages)
    reduction <- as.numeric(cmr$reduction_fraction)
    if (source_age %in% age_labels) {
      for (target_age in target_ages) {
        if (target_age %in% age_labels) mat[target_age, source_age] <- mat[target_age, source_age] * (1 - reduction)
      }
    }
    out$contact_matrix$rows <- contact_rows_from_matrix_r(mat)
    out$contact_matrix$reciprocity_correction$enabled <- FALSE
    out$metadata <- out$metadata %||% list()
    out$metadata$contact_matrix_directional_intervention <- TRUE
  }
  out
}

FALLBACK_TIMELINESS_TARGET_DISTRIBUTION_R <- list(
  infant_3_11m = c(dose1_recent = 0.10, dose2_recent = 0.30, recent = 0.60),
  child_1_4y = c(recent = 0.80, waned = 0.20),
  child_5_9y = c(recent = 0.65, waned = 0.35)
)

parse_schedule_age_months_r <- function(age_pattern) {
  if (is.null(age_pattern) || !nzchar(age_pattern)) return(numeric())
  tokens <- strsplit(as.character(age_pattern), ";", fixed = TRUE)[[1]]
  ages <- numeric()
  for (token in tokens) {
    token <- trimws(toupper(token))
    if (!nzchar(token)) next
    parts <- strsplit(token, "-", fixed = TRUE)[[1]]
    parsed <- numeric()
    last_unit <- ""
    for (part in parts) {
      match <- regexec("^([WMY]?)([0-9]+(?:\\.[0-9]+)?)$", trimws(part), perl = TRUE)
      bits <- regmatches(trimws(part), match)[[1]]
      if (!length(bits)) next
      unit <- bits[[2]]
      if (!nzchar(unit)) unit <- last_unit
      last_unit <- unit
      value <- as.numeric(bits[[3]])
      parsed <- c(parsed, switch(unit, W = value * 7 / 30.4375, M = value, Y = value * 12, value))
    }
    if (length(parsed)) ages <- c(ages, min(parsed))
  }
  sort(ages)
}

timeliness_target_distribution_from_schedule_r <- function(config) {
  schedule_months <- parse_schedule_age_months_r((config$metadata %||% list())$routine_age_pattern %||% "")
  if (!length(schedule_months)) return(FALLBACK_TIMELINESS_TARGET_DISTRIBUTION_R)
  bins <- list(
    infant_3_11m = c(3, 12),
    child_1_4y = c(12, 60),
    child_5_9y = c(60, 120)
  )
  out <- list()
  for (age_group in names(bins)) {
    age_start <- bins[[age_group]][[1]]
    age_end <- bins[[age_group]][[2]]
    midpoints <- seq(age_start, age_end, length.out = 240)
    counts <- numeric()
    for (age_month in midpoints) {
      due <- schedule_months + 1
      due <- due[due <= age_month]
      if (!length(due)) next
      dose_count <- length(due)
      origin <- if (dose_count <= 1) {
        "dose1_recent"
      } else if (dose_count == 2) {
        "dose2_recent"
      } else if (age_month - tail(due, 1) <= 48) {
        "recent"
      } else {
        "waned"
      }
      current <- if (origin %in% names(counts)) counts[[origin]] else 0
      counts[[origin]] <- current + 1
    }
    out[[age_group]] <- if (length(counts) && sum(counts) > 0) counts / sum(counts) else FALLBACK_TIMELINESS_TARGET_DISTRIBUTION_R[[age_group]]
  }
  out
}

apply_timeliness_r <- function(config) {
  out <- config
  out$routine_vaccination$target_relaxation_rate_per_year <- max(
    as.numeric(out$routine_vaccination$target_relaxation_rate_per_year %||% 0),
    6
  )
  out$routine_vaccination$max_daily_flow_fraction <- max(
    as.numeric(out$routine_vaccination$max_daily_flow_fraction %||% 0),
    0.03
  )
  targets <- out$routine_vaccination$target_origin_distribution_by_age %||% list()
  schedule_targets <- timeliness_target_distribution_from_schedule_r(out)
  for (age in names(schedule_targets)) targets[[age]] <- as.list(schedule_targets[[age]])
  out$routine_vaccination$target_origin_distribution_by_age <- targets
  out$metadata <- out$metadata %||% list()
  out$metadata$timeliness_definition <- "schedule_relative"
  out
}

set_interactive_runtime_r <- function(config,
                                      analysis_start_date = "2027-01-01",
                                      analysis_end_date = "2035-12-31",
                                      burn_in_years = 3,
                                      output_time_step = 90,
                                      solver_method = "rk4",
                                      rk4_dt_analysis = 30,
                                      rk4_dt_burnin = 30,
                                      rtol = 1e-4,
                                      atol = 1e-6) {
  out <- config
  out$calendar$analysis_start_date <- as.character(analysis_start_date)
  out$calendar$analysis_end_date <- as.character(analysis_end_date)
  out$simulation$burn_in_years <- as.numeric(burn_in_years)
  out$simulation$output_time_step <- as.numeric(output_time_step)
  out$simulation$solver_method <- as.character(solver_method)
  out$simulation$rk4_dt_analysis <- as.numeric(rk4_dt_analysis)
  out$simulation$rk4_dt_burnin <- as.numeric(rk4_dt_burnin)
  out$simulation$rtol <- as.numeric(rtol)
  out$simulation$atol <- as.numeric(atol)
  out$demography$mode <- "fixed_population_profile"
  resolve_calendar_horizon_r(out)
}

make_strategy_config_r <- function(base_config, strategy, configs = load_app_configs_r()) {
  out <- base_config
  vaccine_name <- base_config$metadata$vaccine_scenario %||% configs$baseline$baseline_vaccine_scenario %||% "symptom_protective"
  if (strategy == "current") {
    return(list(config = out, vaccine_scenario = vaccine_name, intervention = "current"))
  }
  if (strategy == "timeliness_only") {
    return(list(config = apply_timeliness_r(out), vaccine_scenario = vaccine_name, intervention = "timeliness_only"))
  }
  intervention <- configs$interventions[[strategy]]
  if (is.null(intervention)) stop("Unknown strategy: ", strategy, call. = FALSE)
  if (!is.null(intervention$vaccine_scenario)) {
    vaccine_name <- intervention$vaccine_scenario
    out <- apply_vaccine_scenario_r(out, configs$vaccines[[vaccine_name]])
  }
  out <- apply_intervention_definition_r(out, intervention)
  list(config = out, vaccine_scenario = vaccine_name, intervention = strategy)
}

run_strategy_set_r <- function(base_config,
                               strategies = DEFAULT_STRATEGIES_R,
                               configs = load_app_configs_r(),
                               progress = NULL,
                               n_jobs = 1) {
  run_one <- function(i) {
    strategy <- strategies[[i]]
    if (!is.null(progress)) progress(sprintf("Running %s", STRATEGY_LABELS_R[[strategy]] %||% strategy), i / length(strategies))
    built <- make_strategy_config_r(base_config, strategy, configs)
    meta <- list(
      country = base_config$country %||% "Custom",
      strategy_label = STRATEGY_LABELS_R[[strategy]] %||% strategy,
      implementation_intensity = as.numeric(STRATEGY_INTENSITY_R[[strategy]] %||% 1)
    )
    run_prepared_config_r(
      built$config,
      analysis = "interactive_strategy",
      scenario = strategy,
      vaccine_scenario = built$vaccine_scenario,
      resistance_scenario = base_config$metadata$resistance_scenario %||% "country_timeline",
      intervention = built$intervention,
      metadata = meta
    )
  }
  workers <- max(1, min(as.integer(n_jobs %||% 1), length(strategies)))
  if (workers > 1) {
    if (.Platform$OS.type == "windows") {
      stop("Parallel strategy execution is not available on Windows in this app; set workers to 1.", call. = FALSE)
    }
    progress <- NULL
    results <- parallel::mclapply(seq_along(strategies), run_one, mc.cores = workers)
  } else {
    results <- lapply(seq_along(strategies), run_one)
  }
  require_namespace_r("dplyr")
  timeseries <- dplyr::bind_rows(lapply(results, `[[`, "timeseries"))
  summary <- dplyr::bind_rows(lapply(results, `[[`, "summary"))
  summary <- add_relative_reductions_r(summary, "current")
  list(timeseries = timeseries, summary = summary)
}

quick_calibrate_beta_r <- function(config,
                                   target_reported_incidence_per_100k,
                                   configs = load_app_configs_r()) {
  target <- as.numeric(target_reported_incidence_per_100k)
  if (!is.finite(target) || target <= 0) return(config)
  multipliers <- c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4)
  base_beta <- as.numeric(config$transmission$beta_S)
  scores <- rep(Inf, length(multipliers))
  for (i in seq_along(multipliers)) {
    trial <- config
    trial$transmission$beta_S <- base_beta * multipliers[[i]]
    trial <- set_interactive_runtime_r(
      trial,
      analysis_start_date = trial$calendar$analysis_start_date,
      analysis_end_date = as.character(as.Date(trial$calendar$analysis_start_date) + 365),
      burn_in_years = min(1, as.numeric(trial$simulation$burn_in_years %||% 1)),
      output_time_step = 60
    )
    trial_result <- tryCatch(
      run_prepared_config_r(trial, scenario = "calibration_probe", metadata = list(country = trial$country %||% "Custom")),
      error = function(e) NULL
    )
    if (is.null(trial_result)) next
    modeled <- as.numeric(trial_result$summary$annualized_reported_cases_per_100k[[1]])
    scores[[i]] <- abs(log((modeled + 1e-6) / target))
  }
  best <- multipliers[[which.min(scores)]]
  out <- config
  out$transmission$beta_S <- base_beta * best
  out$metadata <- out$metadata %||% list()
  out$metadata$quick_calibration_target_reported_incidence_per_100k <- target
  out$metadata$quick_calibration_beta_multiplier <- best
  out
}
