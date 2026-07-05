RATE_TO_COUNT_COLUMNS_R <- c(
  symptomatic_case_rate_per_day = "symptomatic_cases",
  asymptomatic_infection_rate_per_day = "asymptomatic_infections",
  total_infection_rate_per_day = "total_infections",
  reported_case_rate_per_day = "reported_cases",
  infant_case_rate_per_day = "infant_cases",
  infant_infection_rate_per_day = "infant_infections",
  child_1_9_case_rate_per_day = "child_1_9_cases",
  adolescent_case_rate_per_day = "adolescent_cases",
  child_adolescent_case_rate_per_day = "child_adolescent_cases",
  child_adolescent_infection_rate_per_day = "child_adolescent_infections",
  child_adolescent_reported_case_rate_per_day = "child_adolescent_reported_cases",
  hospitalization_rate_per_day = "hospitalizations",
  infant_hospitalization_rate_per_day = "infant_hospitalizations",
  child_adolescent_hospitalization_rate_per_day = "child_adolescent_hospitalizations",
  mortality_rate_per_day = "deaths",
  infant_mortality_rate_per_day = "infant_deaths",
  child_adolescent_mortality_rate_per_day = "child_adolescent_deaths",
  treated_case_rate_per_day = "treated_cases",
  PEP_FOI_reduction_case_proxy_rate_per_day = "PEP_FOI_reduction_case_proxy",
  PEP_averted_case_rate_per_day = "PEP_averted_cases"
)

AGE_HOSPITALIZATION_PROBABILITY_R <- c(
  infant_0_2m = 0.60,
  infant_3_11m = 0.25,
  child_1_4y = 0.02,
  child_5_9y = 0.003,
  adolescent_10_17y = 0.001,
  young_adult_18_39y = 0.0008,
  middle_adult_40_64y = 0.002,
  elderly_65plus = 0.008
)

AGE_DEATH_GIVEN_HOSPITALIZATION_R <- c(
  infant_0_2m = 0.010,
  infant_3_11m = 0.002,
  child_1_4y = 0.0003,
  child_5_9y = 0.0001,
  adolescent_10_17y = 0.00005,
  young_adult_18_39y = 0.00005,
  middle_adult_40_64y = 0.0003,
  elderly_65plus = 0.002
)

ORIGIN_VE_HOSPITALIZATION_R <- c(
  unvaccinated = 0,
  maternal = 0.72,
  dose1_recent = 0.50,
  dose1_waned = 0.20,
  dose2_recent = 0.65,
  dose2_waned = 0.30,
  recent = 0.75,
  waned = 0.35
)

ORIGIN_VE_DEATH_R <- c(
  unvaccinated = 0,
  maternal = 0.80,
  dose1_recent = 0.60,
  dose1_waned = 0.25,
  dose2_recent = 0.70,
  dose2_waned = 0.35,
  recent = 0.80,
  waned = 0.40
)

active_resistant_fraction_r <- function(y, index) {
  state <- pmax(state_matrix_r(y, index), 0)
  resistant <- sum(state[, strain_state_names_r("R"), drop = FALSE])
  sensitive <- sum(state[, strain_state_names_r("S"), drop = FALSE])
  total <- resistant + sensitive
  if (total > 0) resistant / total else 0
}

compute_severity_outcomes_r <- function(symptomatic_by_origin, age_group, maternal_coverage = 0,
                                        residual_maternal_protection_for_unvaccinated = FALSE) {
  base_hosp <- AGE_HOSPITALIZATION_PROBABILITY_R[[age_group]] %||% 0.001
  base_death <- AGE_DEATH_GIVEN_HOSPITALIZATION_R[[age_group]] %||% 0.0005
  is_infant <- age_group %in% MODEL_INFANT_AGE_GROUPS
  residual_ve_hosp <- if (is_infant && residual_maternal_protection_for_unvaccinated) 0.40 * maternal_coverage else 0
  residual_ve_death <- if (is_infant && residual_maternal_protection_for_unvaccinated) 0.50 * maternal_coverage else 0
  total_hosp <- 0
  total_death <- 0
  total_sym <- 0
  for (origin in names(symptomatic_by_origin)) {
    sym_rate <- as.numeric(symptomatic_by_origin[[origin]])
    if (sym_rate <= 0) next
    ve_hosp <- ORIGIN_VE_HOSPITALIZATION_R[[origin]] %||% 0
    ve_death <- ORIGIN_VE_DEATH_R[[origin]] %||% 0
    if (origin == "unvaccinated" && is_infant) {
      ve_hosp <- max(ve_hosp, residual_ve_hosp)
      ve_death <- max(ve_death, residual_ve_death)
    }
    hosp_rate <- sym_rate * base_hosp * (1 - ve_hosp)
    death_rate <- hosp_rate * base_death * (1 - ve_death)
    total_hosp <- total_hosp + hosp_rate
    total_death <- total_death + death_rate
    total_sym <- total_sym + sym_rate
  }
  c(hospitalizations = total_hosp, deaths = total_death, symptomatic = total_sym)
}

daily_metrics_r <- function(t, y, params, index) {
  state <- pmax(state_matrix_r(y, index), 0)
  foi <- compute_force_of_infection_r(t, y, params, index)
  active_resistance <- active_resistant_fraction_r(y, index)
  reporting_rate <- params$reporting_rate
  current_age_population <- rowSums(state)
  current_total_population <- sum(current_age_population)
  calendar_date <- calendar_date_at_r(params, t)
  calendar_year <- as.integer(format(calendar_date, "%Y"))

  infection_flows <- list(S = list(), R = list())
  pep_averted <- list(S = list(), R = list())
  for (strain in MODEL_STRAINS) {
    lam <- foi[[paste0("lambda_", strain)]]
    base_lam <- foi[[paste0("lambda_", strain, "_base")]]
    for (origin in MODEL_VACCINE_ORIGINS) {
      pool <- state[, susceptible_name_r(origin)]
      infection_flows[[strain]][[origin]] <- lam * params$origin_susceptibility[[origin]] * pool
      base_flow <- base_lam * params$origin_susceptibility[[origin]] * pool
      pep_averted[[strain]][[origin]] <- pmax(base_flow - infection_flows[[strain]][[origin]], 0)
    }
  }

  base_gamma_sym <- params$rates$recovery_symptomatic
  base_gamma_asym <- params$rates$recovery_asymptomatic
  gamma_t_s <- treated_recovery_rate_r(base_gamma_sym, params$treatment, "S")
  gamma_t_r <- treated_recovery_rate_r(base_gamma_sym, params$treatment, "R")
  tr_sym_by_age <- as.numeric(params$treatment$treatment_rate_symptomatic) * params$diagnosis_probability
  tr_asym_by_age <- as.numeric(params$treatment$treatment_rate_asymptomatic) * params$diagnosis_probability
  residual_maternal <- isTRUE((params$raw$severity_model %||% list())$residual_maternal_protection_for_unvaccinated_origin)
  maternal_coverage <- as.numeric((params$demography$birth_entry %||% list())$V %||% 0)

  rows <- vector("list", index$n_age * length(MODEL_STRAINS))
  row_i <- 1
  for (age_idx in seq_len(index$n_age)) {
    age <- index$age_groups[[age_idx]]
    age_population <- current_age_population[[age_idx]]
    for (strain in MODEL_STRAINS) {
      strain_label <- if (strain == "S") "sensitive" else "resistant"
      sym_cases <- asym_infections <- total_infections <- treated_cases <- recoveries <- pep_cases <- 0
      vaccinated_origin_infections <- waned_origin_infections <- maternal_origin_infections <- 0
      dose1_origin_infections <- dose2_origin_infections <- dose3plus_origin_infections <- 0
      sym_by_origin <- setNames(rep(0, length(MODEL_VACCINE_ORIGINS)), MODEL_VACCINE_ORIGINS)

      for (origin in MODEL_VACCINE_ORIGINS) {
        flow <- infection_flows[[strain]][[origin]][[age_idx]]
        p_sym <- params$origin_symptomatic_prob[origin, age_idx]
        origin_sym <- flow * p_sym
        sym_cases <- sym_cases + origin_sym
        sym_by_origin[[origin]] <- sym_by_origin[[origin]] + origin_sym
        asym_infections <- asym_infections + flow * (1 - p_sym)
        total_infections <- total_infections + flow
        pep_cases <- pep_cases + pep_averted[[strain]][[origin]][[age_idx]] * p_sym
        if (origin != "unvaccinated") vaccinated_origin_infections <- vaccinated_origin_infections + flow
        if (origin_is_waned_r(origin)) waned_origin_infections <- waned_origin_infections + flow
        category <- origin_dose_category_r(origin)
        if (category == "maternal") maternal_origin_infections <- maternal_origin_infections + flow
        if (category == "dose1") dose1_origin_infections <- dose1_origin_infections + flow
        if (category == "dose2") dose2_origin_infections <- dose2_origin_infections + flow
        if (category == "dose3plus") dose3plus_origin_infections <- dose3plus_origin_infections + flow

        recovery_multiplier <- params$origin_recovery_mult[[origin]]
        gamma_sym <- base_gamma_sym * recovery_multiplier
        gamma_asym <- base_gamma_asym * recovery_multiplier
        gamma_t <- if (strain == "S") gamma_t_s else gamma_t_r
        gamma_t <- gamma_t * recovery_multiplier
        i_sym <- infectious_name_r(strain, "sym", origin)
        i_asym <- infectious_name_r(strain, "asym", origin)
        treated <- treated_name_r(strain, origin)
        treated_cases <- treated_cases +
          tr_sym_by_age[[age_idx]] * state[age_idx, i_sym] +
          tr_asym_by_age[[age_idx]] * state[age_idx, i_asym]
        recoveries <- recoveries +
          gamma_sym * state[age_idx, i_sym] +
          gamma_asym * state[age_idx, i_asym] +
          gamma_t * state[age_idx, treated]
      }

      severity <- compute_severity_outcomes_r(sym_by_origin, age, maternal_coverage, residual_maternal)
      reported_cases <- sym_cases * reporting_rate[[age_idx]]
      is_infant <- age %in% MODEL_INFANT_AGE_GROUPS
      is_child_1_9 <- age %in% MODEL_CHILD_1_9_AGE_GROUPS
      is_adolescent <- age %in% MODEL_ADOLESCENT_AGE_GROUPS
      is_child_adolescent <- age %in% MODEL_CHILD_ADOLESCENT_AGE_GROUPS

      rows[[row_i]] <- data.frame(
        time = as.numeric(t),
        calendar_date = as.character(calendar_date),
        calendar_year = calendar_year,
        age_group = age,
        strain = strain_label,
        analysis = params$analysis,
        scenario = params$scenario,
        vaccine_scenario = params$vaccine_scenario,
        resistance_scenario = params$resistance_scenario,
        intervention = params$intervention,
        VE_sus = as.numeric(params$vaccine$VE_sus),
        VE_sym = as.numeric(params$vaccine$VE_sym),
        VE_inf = as.numeric(params$vaccine$VE_inf),
        VE_dur = as.numeric(params$vaccine$VE_dur),
        initial_resistance_prevalence = as.numeric(params$initial$initial_resistance_prevalence),
        target_resistance_prevalence_at_analysis_start = target_resistance_prevalence_r(params),
        active_resistant_fraction = active_resistance,
        fitness_R = as.numeric(params$transmission$fitness_R %||% 1),
        pep_coverage_dynamic = as.numeric(foi$pep_coverage),
        population = age_population,
        total_population = current_total_population,
        symptomatic_case_rate_per_day = sym_cases,
        asymptomatic_infection_rate_per_day = asym_infections,
        total_infection_rate_per_day = total_infections,
        reported_case_rate_per_day = reported_cases,
        infant_case_rate_per_day = ifelse(is_infant, sym_cases, 0),
        infant_infection_rate_per_day = ifelse(is_infant, total_infections, 0),
        child_1_9_case_rate_per_day = ifelse(is_child_1_9, sym_cases, 0),
        adolescent_case_rate_per_day = ifelse(is_adolescent, sym_cases, 0),
        child_adolescent_case_rate_per_day = ifelse(is_child_adolescent, sym_cases, 0),
        child_adolescent_infection_rate_per_day = ifelse(is_child_adolescent, total_infections, 0),
        child_adolescent_reported_case_rate_per_day = ifelse(is_child_adolescent, reported_cases, 0),
        hospitalization_rate_per_day = severity[["hospitalizations"]],
        infant_hospitalization_rate_per_day = ifelse(is_infant, severity[["hospitalizations"]], 0),
        child_adolescent_hospitalization_rate_per_day = ifelse(is_child_adolescent, severity[["hospitalizations"]], 0),
        mortality_rate_per_day = severity[["deaths"]],
        infant_mortality_rate_per_day = ifelse(is_infant, severity[["deaths"]], 0),
        child_adolescent_mortality_rate_per_day = ifelse(is_child_adolescent, severity[["deaths"]], 0),
        treated_case_rate_per_day = treated_cases,
        PEP_FOI_reduction_case_proxy_rate_per_day = pep_cases,
        PEP_averted_case_rate_per_day = pep_cases,
        infection_to_recovery_rate_ratio = total_infections / max(recoveries, 1e-9),
        vaccinated_origin_infection_share = ifelse(total_infections > 0, vaccinated_origin_infections / total_infections, 0),
        waned_origin_infection_share = ifelse(total_infections > 0, waned_origin_infections / total_infections, 0),
        maternal_origin_infection_share = ifelse(total_infections > 0, maternal_origin_infections / total_infections, 0),
        dose1_origin_infection_share = ifelse(total_infections > 0, dose1_origin_infections / total_infections, 0),
        dose2_origin_infection_share = ifelse(total_infections > 0, dose2_origin_infections / total_infections, 0),
        dose3plus_origin_infection_share = ifelse(total_infections > 0, dose3plus_origin_infections / total_infections, 0),
        stringsAsFactors = FALSE
      )
      row_i <- row_i + 1
    }
  }
  do.call(rbind, rows)
}

add_interval_counts_r <- function(df) {
  require_namespace_r("dplyr")
  out <- df |>
    dplyr::arrange(.data$analysis, .data$scenario, .data$age_group, .data$strain, .data$time) |>
    dplyr::group_by(.data$analysis, .data$scenario, .data$age_group, .data$strain) |>
    dplyr::mutate(.dt = .data$time - dplyr::lag(.data$time, default = dplyr::first(.data$time))) |>
    dplyr::ungroup()
  for (rate_col in names(RATE_TO_COUNT_COLUMNS_R)) {
    count_col <- RATE_TO_COUNT_COLUMNS_R[[rate_col]]
    out <- out |>
      dplyr::group_by(.data$analysis, .data$scenario, .data$age_group, .data$strain) |>
      dplyr::mutate(
        "{count_col}" := 0.5 * (.data[[rate_col]] + dplyr::lag(.data[[rate_col]], default = dplyr::first(.data[[rate_col]]))) * .data$.dt
      ) |>
      dplyr::ungroup()
  }
  dplyr::select(out, -".dt")
}

compute_timeseries_r <- function(solution, params, index) {
  solution_df <- as.data.frame(solution)
  rows <- vector("list", nrow(solution_df))
  for (i in seq_len(nrow(solution_df))) {
    rows[[i]] <- daily_metrics_r(solution_df$time[[i]], as.numeric(solution_df[i, -1]), params, index)
  }
  df <- do.call(rbind, rows)
  df <- add_interval_counts_r(df)

  require_namespace_r("dplyr")
  resistant <- df |>
    dplyr::mutate(resistant_infections_tmp = ifelse(.data$strain == "resistant", .data$total_infections, 0)) |>
    dplyr::group_by(.data$analysis, .data$scenario, .data$time, .data$age_group) |>
    dplyr::summarise(resistant_infections_tmp = sum(.data$resistant_infections_tmp), all_infections = sum(.data$total_infections), .groups = "drop")
  df <- dplyr::left_join(df, resistant, by = c("analysis", "scenario", "time", "age_group"))
  df$resistant_fraction <- ifelse(df$all_infections > 0, df$resistant_infections_tmp / df$all_infections, 0)
  df$resistant_infections_tmp <- NULL
  df$all_infections <- NULL
  df <- df |>
    dplyr::arrange(.data$analysis, .data$scenario, .data$age_group, .data$strain, .data$time) |>
    dplyr::group_by(.data$analysis, .data$scenario, .data$age_group, .data$strain) |>
    dplyr::mutate(
      cumulative_cases = cumsum(.data$symptomatic_cases),
      cumulative_reported_cases = cumsum(.data$reported_cases),
      cumulative_infections = cumsum(.data$total_infections)
    ) |>
    dplyr::ungroup()
  as.data.frame(df)
}

sum_col_r <- function(df, col) {
  if (!col %in% names(df)) return(0)
  sum(df[[col]], na.rm = TRUE)
}

population_for_age_groups_r <- function(age_population, groups) {
  sum(age_population[names(age_population) %in% groups], na.rm = TRUE)
}

summarize_timeseries_r <- function(df) {
  require_namespace_r("dplyr")
  groups <- c("analysis", "scenario", "vaccine_scenario", "resistance_scenario", "intervention")
  if ("country" %in% names(df)) groups <- c(groups, "country")
  grouped <- split(df, interaction(df[groups], drop = TRUE), drop = TRUE)
  rows <- lapply(grouped, function(group) {
    age_population <- tapply(group$population, group$age_group, mean, na.rm = TRUE)
    total_population <- mean(group$total_population, na.rm = TRUE)
    infant_population <- population_for_age_groups_r(age_population, MODEL_INFANT_AGE_GROUPS)
    child_1_9_population <- population_for_age_groups_r(age_population, MODEL_CHILD_1_9_AGE_GROUPS)
    adolescent_population <- population_for_age_groups_r(age_population, MODEL_ADOLESCENT_AGE_GROUPS)
    child_adolescent_population <- population_for_age_groups_r(age_population, MODEL_CHILD_ADOLESCENT_AGE_GROUPS)
    duration_days <- max(group$time, na.rm = TRUE) - min(group$time, na.rm = TRUE)
    duration_years <- max(duration_days / 365, 1 / 365)
    total_infections <- sum_col_r(group, "total_infections")
    total_symptomatic <- sum_col_r(group, "symptomatic_cases")
    total_reported <- sum_col_r(group, "reported_cases")
    resistant_infections <- sum(group$total_infections[group$strain == "resistant"], na.rm = TRUE)
    first <- group[1, , drop = FALSE]
    data.frame(
      analysis = first$analysis,
      scenario = first$scenario,
      vaccine_scenario = first$vaccine_scenario,
      resistance_scenario = first$resistance_scenario,
      intervention = first$intervention,
      country = if ("country" %in% names(group)) first$country else NA_character_,
      calendar_start_date = min(group$calendar_date),
      calendar_end_date = max(group$calendar_date),
      total_symptomatic_cases = total_symptomatic,
      total_infections = total_infections,
      total_reported_cases = total_reported,
      total_infant_cases = sum_col_r(group, "infant_cases"),
      total_infant_infections = sum_col_r(group, "infant_infections"),
      total_child_1_9_cases = sum_col_r(group, "child_1_9_cases"),
      total_adolescent_cases = sum_col_r(group, "adolescent_cases"),
      total_child_adolescent_cases = sum_col_r(group, "child_adolescent_cases"),
      total_child_adolescent_infections = sum_col_r(group, "child_adolescent_infections"),
      total_child_adolescent_reported_cases = sum_col_r(group, "child_adolescent_reported_cases"),
      total_hospitalizations = sum_col_r(group, "hospitalizations"),
      total_child_adolescent_hospitalizations = sum_col_r(group, "child_adolescent_hospitalizations"),
      total_deaths = sum_col_r(group, "deaths"),
      total_infant_deaths = sum_col_r(group, "infant_deaths"),
      total_child_adolescent_deaths = sum_col_r(group, "child_adolescent_deaths"),
      resistant_infections = resistant_infections,
      resistant_fraction = ifelse(total_infections > 0, resistant_infections / total_infections, 0),
      resistant_fraction_start = group$active_resistant_fraction[which.min(group$time)],
      resistant_fraction_end = group$active_resistant_fraction[which.max(group$time)],
      total_population = total_population,
      infant_population = infant_population,
      child_1_9_population = child_1_9_population,
      adolescent_population = adolescent_population,
      child_adolescent_population = child_adolescent_population,
      analysis_years = duration_years,
      annualized_infections_per_100k = total_infections / max(duration_years * total_population, 1e-9) * 100000,
      annualized_reported_cases_per_100k = total_reported / max(duration_years * total_population, 1e-9) * 100000,
      annualized_infant_cases_per_100k = sum_col_r(group, "infant_cases") / max(duration_years * infant_population, 1e-9) * 100000,
      annualized_child_1_9_cases_per_100k = sum_col_r(group, "child_1_9_cases") / max(duration_years * child_1_9_population, 1e-9) * 100000,
      annualized_adolescent_cases_per_100k = sum_col_r(group, "adolescent_cases") / max(duration_years * adolescent_population, 1e-9) * 100000,
      annualized_child_adolescent_cases_per_100k = sum_col_r(group, "child_adolescent_cases") / max(duration_years * child_adolescent_population, 1e-9) * 100000,
      annualized_resistant_infections_per_100k = resistant_infections / max(duration_years * total_population, 1e-9) * 100000,
      annualized_deaths_per_million = sum_col_r(group, "deaths") / max(duration_years * total_population, 1e-9) * 1000000,
      annualized_hospitalizations_per_100k = sum_col_r(group, "hospitalizations") / max(duration_years * total_population, 1e-9) * 100000,
      treated_cases = sum_col_r(group, "treated_cases"),
      PEP_averted_cases = sum_col_r(group, "PEP_averted_cases"),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

add_relative_reductions_r <- function(summary, reference_scenario = "current") {
  out <- summary
  mapping <- c(
    relative_reduction_infant_cases = "total_infant_cases",
    relative_reduction_child_adolescent_cases = "total_child_adolescent_cases",
    relative_reduction_total_infections = "total_infections",
    relative_reduction_reported_cases = "total_reported_cases",
    relative_reduction_resistant_infections = "resistant_infections",
    relative_reduction_deaths = "total_deaths",
    relative_reduction_hospitalizations = "total_hospitalizations"
  )
  for (new_col in names(mapping)) out[[new_col]] <- NA_real_
  split_keys <- if ("country" %in% names(out) && any(!is.na(out$country))) out$country else rep("all", nrow(out))
  for (key in unique(split_keys)) {
    idx <- which(split_keys == key)
    ref <- idx[out$scenario[idx] == reference_scenario]
    if (!length(ref)) next
    ref <- ref[[1]]
    for (new_col in names(mapping)) {
      source_col <- mapping[[new_col]]
      denom <- as.numeric(out[[source_col]][[ref]])
      out[[new_col]][idx] <- if (denom > 0) {
        1 - as.numeric(out[[source_col]][idx]) / denom
      } else {
        rep(NA_real_, length(idx))
      }
      zero_idx <- idx[abs(denom) <= 1e-12 & abs(as.numeric(out[[source_col]][idx])) <= 1e-12]
      out[[new_col]][zero_idx] <- 0
    }
  }
  out$relative_reduction_vs_baseline <- out$relative_reduction_total_infections
  out
}
