## Extended Data Figure 12 data preparation -----------------------------------

load_extended_data_figure_12_baseline <- function() {
  read_model_table(model_path("outputs", "summaries", "country_scenarios_summary")) %>%
    add_country_label()
}

extended_data_figure_12_with_burden_order <- function(df, baseline) {
  baseline_order <- baseline %>%
    arrange(desc(annualized_infant_cases_per_100k)) %>%
    pull(country_label) %>%
    as.character()

  df %>%
    mutate(country_burden_order = factor(as.character(country_label), levels = rev(baseline_order)))
}

load_extended_data_figure_12_inputs <- function() {
  baseline <- load_extended_data_figure_12_baseline()

  fitness_summary <- read_model_table_optional(model_path("outputs", "summaries", "fitness_resistance_grid_summary"))
  if (nrow(fitness_summary) > 0) {
    fitness_summary <- fitness_summary %>%
      add_country_label() %>%
      mutate(
        grid_fitness_R = as.numeric(grid_fitness_R),
        grid_VE_inf = as.numeric(grid_VE_inf)
      ) %>%
      extended_data_figure_12_with_burden_order(baseline)
  }

  fitness_benefit_parameter_diagnostics <- read_model_table_optional(
    model_path("outputs", "summaries", "fitness_resistance_grid_posterior_sample_diagnostics")
  )
  if (nrow(fitness_benefit_parameter_diagnostics) > 0) {
    fitness_benefit_parameter_diagnostics <- fitness_benefit_parameter_diagnostics %>%
      add_country_label() %>%
      mutate(
        posterior_draws = as.integer(posterior_draws),
        unique_values = as.integer(unique_values),
        mean = as.numeric(mean),
        sd = as.numeric(sd),
        coefficient_of_variation = as.numeric(coefficient_of_variation),
        q025 = as.numeric(q025),
        median = as.numeric(median),
        q975 = as.numeric(q975),
        varies_within_country = as.logical(varies_within_country),
        grid_override_in_fig3d = as.logical(grid_override_in_fig3d),
        uncertainty_source = as.character(uncertainty_source)
      ) %>%
      extended_data_figure_12_with_burden_order(baseline)
  }

  fitness_benefit_psa_summary <- read_model_table_optional(
    model_path("outputs", "tables", "fitness_resistance_grid_psa_benefit_summary")
  )
  if (nrow(fitness_benefit_psa_summary) > 0) {
    fitness_benefit_psa_summary <- fitness_benefit_psa_summary %>%
      add_country_label() %>%
      mutate(
        fitness_group = factor(
          fitness_group,
          levels = c("Fitness cost (0.85)", "Neutral (1.00)", "Advantage (1.10)")
        ),
        grid_fitness_R = as.numeric(grid_fitness_R),
        low_grid_VE_inf = as.numeric(low_grid_VE_inf),
        high_grid_VE_inf = as.numeric(high_grid_VE_inf),
        median_relative_benefit = as.numeric(median_relative_benefit),
        q025_relative_benefit = as.numeric(q025_relative_benefit),
        q975_relative_benefit = as.numeric(q975_relative_benefit),
        psa_samples = as.integer(psa_samples),
        uncertainty_source = as.character(uncertainty_source),
        uncertainty_scope = if ("uncertainty_scope" %in% names(.)) as.character(uncertainty_scope) else NA_character_
      ) %>%
      extended_data_figure_12_with_burden_order(baseline)
  }

  list(
    baseline = baseline,
    fitness_summary = fitness_summary,
    fitness_benefit_parameter_diagnostics = fitness_benefit_parameter_diagnostics,
    fitness_benefit_psa_summary = fitness_benefit_psa_summary
  )
}

require_extended_data_figure_12_inputs <- function(inputs) {
  if (nrow(inputs$fitness_summary) == 0) {
    stop(
      "eFigure 12 requires outputs/summaries/fitness_resistance_grid_summary. ",
      "Run src_python.simulation.run_fitness_grid before plotting.",
      call. = FALSE
    )
  }
  if (nrow(inputs$fitness_benefit_parameter_diagnostics) == 0) {
    stop(
      "eFigure 12 requires outputs/summaries/fitness_resistance_grid_posterior_sample_diagnostics. ",
      "Run src_python.simulation.run_fitness_grid with posterior benefit diagnostics first.",
      call. = FALSE
    )
  }
  if (nrow(inputs$fitness_benefit_psa_summary) == 0) {
    stop(
      "eFigure 12 requires outputs/tables/fitness_resistance_grid_psa_benefit_summary. ",
      "Run src_python.simulation.run_fitness_grid with --psa-benefit-samples before plotting.",
      call. = FALSE
    )
  }
  invisible(inputs)
}

smooth_extended_data_figure_12_surface <- function(surface, value_col, transform = c("identity", "log10", "logit"), grid_n = 180) {
  transform <- match.arg(transform)
  values <- surface[[value_col]]
  model_values <- switch(
    transform,
    identity = values,
    log10 = log10(pmax(values, 1e-9)),
    logit = qlogis(pmin(pmax(values, 1e-6), 1 - 1e-6))
  )

  model_data <- surface
  model_data$.surface_value <- model_values

  pred_grid <- tidyr::expand_grid(
    grid_fitness_R = seq(
      min(surface$grid_fitness_R, na.rm = TRUE),
      max(surface$grid_fitness_R, na.rm = TRUE),
      length.out = grid_n
    ),
    grid_VE_inf = seq(
      min(surface$grid_VE_inf, na.rm = TRUE),
      max(surface$grid_VE_inf, na.rm = TRUE),
      length.out = grid_n
    )
  )

  loess_fit <- stats::loess(
    .surface_value ~ grid_fitness_R + grid_VE_inf,
    data = model_data,
    span = 0.55,
    degree = 2,
    control = stats::loess.control(surface = "direct")
  )
  predicted <- as.numeric(stats::predict(loess_fit, newdata = pred_grid))

  if (anyNA(predicted)) {
    fallback_fit <- stats::lm(
      .surface_value ~ stats::poly(grid_fitness_R, 2, raw = TRUE) *
        stats::poly(grid_VE_inf, 2, raw = TRUE),
      data = model_data
    )
    missing_predictions <- is.na(predicted)
    predicted[missing_predictions] <- as.numeric(stats::predict(
      fallback_fit,
      newdata = pred_grid[missing_predictions, , drop = FALSE]
    ))
  }

  pred_grid$smoothed_value <- switch(
    transform,
    identity = predicted,
    log10 = 10^predicted,
    logit = plogis(predicted)
  )
  pred_grid
}

prepare_extended_data_figure_12_data <- function(inputs = load_extended_data_figure_12_inputs()) {
  require_extended_data_figure_12_inputs(inputs)

  fitness_summary <- inputs$fitness_summary
  fitness_benefit_parameter_diagnostics <- inputs$fitness_benefit_parameter_diagnostics
  fitness_benefit_psa_summary <- inputs$fitness_benefit_psa_summary

  grid_veinf_levels <- sort(unique(fitness_summary$grid_VE_inf))
  low_grid_veinf <- grid_veinf_levels[[1]]
  high_grid_veinf <- grid_veinf_levels[[length(grid_veinf_levels)]]
  fitness_range <- range(fitness_summary$grid_fitness_R, na.rm = TRUE)
  ve_inf_range <- range(fitness_summary$grid_VE_inf, na.rm = TRUE)
  fitness_breaks <- sort(unique(c(fitness_range, 0.85, 1.00, 1.15)))
  ve_inf_breaks <- sort(unique(c(ve_inf_range, seq(0.15, 0.45, by = 0.15))))

  fitness_surface <- fitness_summary %>%
    group_by(grid_fitness_R, grid_VE_inf) %>%
    summarise(
      median_resistant_end = median(resistant_fraction_end, na.rm = TRUE),
      median_infant_cases = median(annualized_infant_cases_per_100k, na.rm = TRUE),
      median_all_infections = median(annualized_infections_per_100k, na.rm = TRUE),
      .groups = "drop"
    )

  surface_resistant <- smooth_extended_data_figure_12_surface(fitness_surface, "median_resistant_end", "logit") %>%
    mutate(median_resistant_end = pmin(pmax(smoothed_value, 0), 1))

  surface_infant <- smooth_extended_data_figure_12_surface(fitness_surface, "median_infant_cases", "log10") %>%
    mutate(median_infant_cases = pmax(smoothed_value, 1e-6))

  surface_all <- smooth_extended_data_figure_12_surface(fitness_surface, "median_all_infections", "log10") %>%
    mutate(median_all_infections = pmax(smoothed_value, 1e-6))

  diagnostic_parameter_order <- c(
    "beta_S",
    "reporting_multiplier",
    "reporting_trend_end_multiplier",
    "resistance_prevalence",
    "VE_sus",
    "VE_inf",
    "VE_dur",
    "fitness_R",
    "relative_infectiousness_asymptomatic",
    "infectious_duration_asymptomatic",
    "infectious_duration_symptomatic"
  )

  diagnostic_status <- fitness_benefit_parameter_diagnostics %>%
    mutate(
      parameter = factor(parameter, levels = diagnostic_parameter_order),
      parameter_status = case_when(
        grid_override_in_fig3d ~ "Grid override",
        varies_within_country ~ "Posterior varied",
        TRUE ~ "Fixed in selected posterior"
      ),
      parameter_status = factor(
        parameter_status,
        levels = c("Posterior varied", "Grid override", "Fixed in selected posterior")
      )
    )

  grid_benefit_full <- fitness_summary %>%
    filter(grid_VE_inf %in% c(low_grid_veinf, high_grid_veinf)) %>%
    mutate(ve_level = if_else(grid_VE_inf == low_grid_veinf, "ve_low", "ve_high")) %>%
    select(country_burden_order, grid_fitness_R, ve_level, annualized_infant_cases_per_100k) %>%
    pivot_wider(names_from = ve_level, values_from = annualized_infant_cases_per_100k) %>%
    mutate(
      relative_benefit = (ve_low - ve_high) / pmax(ve_low, 1e-9),
      fitness_label = factor(
        format_lancet_fixed(grid_fitness_R, digits = 2),
        levels = format_lancet_fixed(sort(unique(fitness_summary$grid_fitness_R)), digits = 2)
      )
    )

  psa_country_levels <- levels(fitness_benefit_psa_summary$country_burden_order)
  if (is.null(psa_country_levels)) {
    psa_country_levels <- unique(as.character(fitness_benefit_psa_summary$country_burden_order))
  }

  fitness_group_offsets <- c(
    "Fitness cost (0.85)" = -0.22,
    "Neutral (1.00)" = 0,
    "Advantage (1.10)" = 0.22
  )

  fitness_group_colours <- setNames(
    shared_palettes$discrete$ten[c(4, 10, 1)],
    levels(fitness_benefit_psa_summary$fitness_group)
  )

  psa_interval_data <- fitness_benefit_psa_summary %>%
    mutate(
      country_index = as.numeric(factor(country_burden_order, levels = psa_country_levels)),
      y_position = country_index + fitness_group_offsets[as.character(fitness_group)]
    )

  list(
    fitness_summary = fitness_summary,
    fitness_benefit_parameter_diagnostics = fitness_benefit_parameter_diagnostics,
    fitness_benefit_psa_summary = fitness_benefit_psa_summary,
    low_grid_veinf = low_grid_veinf,
    high_grid_veinf = high_grid_veinf,
    fitness_range = fitness_range,
    ve_inf_range = ve_inf_range,
    fitness_breaks = fitness_breaks,
    ve_inf_breaks = ve_inf_breaks,
    infant_case_colourbar_breaks = c(1, 10, 100, 1000, 4000),
    infant_case_contour_breaks = c(1, 10, 30, 100, 300, 1000, 3000, 4000),
    all_infection_colourbar_breaks = c(3, 30, 300, 1000, 3000),
    all_infection_contour_breaks = c(3, 10, 30, 100, 300, 1000, 3000, 4000),
    surface_resistant = surface_resistant,
    surface_infant = surface_infant,
    surface_all = surface_all,
    diagnostic_status = diagnostic_status,
    grid_benefit_full = grid_benefit_full,
    psa_country_levels = psa_country_levels,
    fitness_group_colours = fitness_group_colours,
    psa_interval_data = psa_interval_data
  )
}
