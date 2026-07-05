args <- commandArgs(FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[[1]])) else file.path(getwd(), "scripts_R")
source(file.path(script_dir, "10_shared.R"))

## Extended Data Figure 12: fitness_R x VE_inf stress-test grid. ----------------------
## The grid varies resistant-strain fitness (0.70-1.25) crossed with vaccine
## infectiousness effect (VE_inf 0.05-0.55) across all country profiles.

if (nrow(fitness_summary) == 0) {
  stop(
    "eFigure 12 requires outputs/summaries/fitness_resistance_grid_summary. ",
    "Run src_python.simulation.run_fitness_grid before plotting."
  )
}
if (nrow(fitness_benefit_parameter_diagnostics) == 0) {
  stop(
    "eFigure 12 requires outputs/summaries/fitness_resistance_grid_posterior_sample_diagnostics. ",
    "Run src_python.simulation.run_fitness_grid with posterior benefit diagnostics first."
  )
}
if (nrow(fitness_benefit_psa_summary) == 0) {
  stop(
    "eFigure 12 requires outputs/tables/fitness_resistance_grid_psa_benefit_summary. ",
    "Run src_python.simulation.run_fitness_grid with --psa-benefit-samples before plotting."
  )
}

smooth_surface_grid <- function(surface, value_col, transform = c("identity", "log10", "logit"), grid_n = 180) {
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

grid_veinf_levels <- sort(unique(fitness_summary$grid_VE_inf))
low_grid_veinf <- grid_veinf_levels[[1]]
high_grid_veinf <- grid_veinf_levels[[length(grid_veinf_levels)]]
fitness_range <- range(fitness_summary$grid_fitness_R, na.rm = TRUE)
ve_inf_range <- range(fitness_summary$grid_VE_inf, na.rm = TRUE)
fitness_breaks <- sort(unique(c(fitness_range, 0.85, 1.00, 1.15)))
ve_inf_breaks <- sort(unique(c(ve_inf_range, seq(0.15, 0.45, by = 0.15))))
infant_case_colourbar_breaks <- c(1, 10, 100, 1000, 4000)
infant_case_contour_breaks <- c(1, 10, 30, 100, 300, 1000, 3000, 4000)
all_infection_colourbar_breaks <- c(3, 30, 300, 1000, 3000)
all_infection_contour_breaks <- c(3, 10, 30, 100, 300, 1000, 3000, 4000)

fitness_surface <- fitness_summary %>%
  group_by(grid_fitness_R, grid_VE_inf) %>%
  summarise(
    median_resistant_end = median(resistant_fraction_end, na.rm = TRUE),
    median_infant_cases = median(annualized_infant_cases_per_100k, na.rm = TRUE),
    median_all_infections = median(annualized_infections_per_100k, na.rm = TRUE),
    .groups = "drop"
  )

surface_resistant <- smooth_surface_grid(fitness_surface, "median_resistant_end", "logit") %>%
  mutate(median_resistant_end = pmin(pmax(smoothed_value, 0), 1))

surface_infant <- smooth_surface_grid(fitness_surface, "median_infant_cases", "log10") %>%
  mutate(median_infant_cases = pmax(smoothed_value, 1e-6))

surface_all <- smooth_surface_grid(fitness_surface, "median_all_infections", "log10") %>%
  mutate(median_all_infections = pmax(smoothed_value, 1e-6))

surface_axes <- list(
  scale_x_continuous(
    breaks = fitness_breaks,
    labels = label_lancet_number(accuracy = 0.01),
    expand = expansion(mult = 0)
  ),
  scale_y_continuous(
    breaks = ve_inf_breaks,
    labels = label_lancet_percent(accuracy = 1),
    expand = expansion(mult = 0)
  ),
  coord_cartesian(xlim = fitness_range, ylim = ve_inf_range, expand = FALSE)
)

p_ed12a <- ggplot(surface_resistant, aes(grid_fitness_R, grid_VE_inf)) +
  geom_raster(aes(fill = median_resistant_end), interpolate = TRUE) +
  geom_contour(
    aes(z = median_resistant_end),
    breaks = seq(0.1, 0.9, by = 0.1),
    colour = "white",
    alpha = 0.55,
    linewidth = 0.14
  ) +
  geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
  surface_axes +
  scale_fill_resistance_fraction(
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    guide = guide_lancet_colourbar(barwidth = unit(4.2, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
  ) +
  labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "End resistant\nfraction") +
  theme_lancet_heatmap(base_size = journal_compact_text_size)

p_ed12b <- ggplot(surface_infant, aes(grid_fitness_R, grid_VE_inf)) +
  geom_raster(aes(fill = median_infant_cases), interpolate = TRUE) +
  geom_contour(
    aes(z = median_infant_cases),
    breaks = infant_case_contour_breaks,
    colour = "white",
    alpha = 0.55,
    linewidth = 0.14
  ) +
  geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
  surface_axes +
  scale_fill_infant_burden(
    limits = c(1, 4000),
    breaks = infant_case_colourbar_breaks,
    labels = label_lancet_comma(accuracy = 1),
    trans = "log10",
    oob = scales::squish,
    guide = guide_lancet_colourbar(barwidth = unit(5.0, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
  ) +
  labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "Infant cases\nper 100 000/year") +
  theme_lancet_heatmap(base_size = journal_compact_text_size)

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

diagnostic_parameter_labels <- c(
  beta_S = "beta[S]",
  reporting_multiplier = "'Rpt.'",
  reporting_trend_end_multiplier = "atop('Rpt.','trend')",
  resistance_prevalence = "atop('R','prev.')",
  VE_sus = "VE[sus]",
  VE_inf = "VE[inf]",
  VE_dur = "VE[dur]",
  fitness_R = "italic(f)[R]",
  relative_infectiousness_asymptomatic = "atop('Asympt.','rel.')",
  infectious_duration_asymptomatic = "atop('Asympt.','dur.')",
  infectious_duration_symptomatic = "atop('Sympt.','dur.')"
)

diagnostic_parameter_labeller <- function(values) {
  labels <- diagnostic_parameter_labels[as.character(values)]
  missing_labels <- is.na(labels)
  if (any(missing_labels)) {
    labels[missing_labels] <- vapply(as.character(values[missing_labels]), shQuote, character(1))
  }
  parse(text = unname(labels))
}

diagnostic_status_labels <- c(
  "Posterior varied" = "Varied",
  "Grid override" = "Grid override",
  "Fixed in selected posterior" = "Fixed"
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

p_ed12c <- ggplot(diagnostic_status, aes(parameter, country_burden_order, fill = parameter_status)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  scale_x_discrete(labels = diagnostic_parameter_labeller, expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  scale_fill_manual(
    values = manuscript_parameter_audit_colours,
    breaks = names(manuscript_parameter_audit_colours),
    labels = diagnostic_status_labels,
    guide = guide_legend(nrow = 1, keywidth = unit(0.22, "cm"), keyheight = unit(0.16, "cm"))
  ) +
  labs(x = NULL, y = NULL, fill = "Role") +
  theme_lancet_heatmap(
    base_size = journal_dense_text_size,
    x_angle = 90,
    x_hjust = 1,
    x_vjust = 0.5,
    x_size = 4.5,
    y_size = 5.1
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

p_ed12d <- grid_benefit_full %>%
  ggplot(aes(fitness_label, country_burden_order, fill = relative_benefit)) +
  geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_fill_fraction(
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.50, 0.75, 1.00),
    labels = label_lancet_percent(accuracy = 1),
    oob = scales::squish,
    guide = guide_lancet_colourbar(barwidth = unit(4.8, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
  ) +
  labs(
    x = expression(italic(f)[R]),
    y = NULL,
    fill = paste0("High VE_inf benefit\n(", lancet_percent(high_grid_veinf), " vs ", lancet_percent(low_grid_veinf), ")")
  ) +
  theme_lancet_heatmap(
    base_size = journal_compact_text_size,
    x_angle = 90,
    x_hjust = 1,
    x_vjust = 0.5,
    x_size = 5.2
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

p_ed12e <- ggplot(psa_interval_data, aes(y = y_position, colour = fitness_group)) +
  geom_segment(
    aes(x = q025_relative_benefit, xend = q975_relative_benefit, yend = y_position),
    linewidth = 0.42,
    lineend = "round"
  ) +
  geom_point(aes(x = median_relative_benefit), size = 1.0) +
  scale_x_continuous(
    labels = label_lancet_percent(accuracy = 1),
    breaks = seq(0, 1, by = 0.25),
    limits = c(0, 1),
    oob = scales::squish,
    expand = expansion(mult = c(0, 0.02))
  ) +
  scale_y_continuous(
    breaks = seq_along(psa_country_levels),
    labels = psa_country_levels,
    expand = expansion(add = c(0.45, 0.45))
  ) +
  scale_colour_manual(
    values = fitness_group_colours,
    breaks = names(fitness_group_colours),
    guide = guide_legend(nrow = 1, keywidth = unit(0.28, "cm"), keyheight = unit(0.18, "cm"))
  ) +
  labs(
    x = paste0("High VE_inf benefit (", lancet_percent(high_grid_veinf), " vs ", lancet_percent(low_grid_veinf), ")"),
    y = NULL,
    colour = NULL
  ) +
  theme_lancet_panel(base_size = journal_compact_text_size, show_y_grid = TRUE) +
  theme(
    legend.position = "bottom",
    panel.grid.major.x = element_line(linewidth = lancet_grid_linewidth, colour = lancet_grid_light_colour),
    axis.text.y = element_text(size = 5.4)
  )

p_ed12f <- ggplot(surface_all, aes(grid_fitness_R, grid_VE_inf)) +
  geom_raster(aes(fill = median_all_infections), interpolate = TRUE) +
  geom_contour(
    aes(z = median_all_infections),
    breaks = all_infection_contour_breaks,
    colour = "white",
    alpha = 0.55,
    linewidth = 0.14
  ) +
  geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
  surface_axes +
  scale_fill_infant_burden(
    limits = c(3, 4000),
    breaks = all_infection_colourbar_breaks,
    labels = label_lancet_comma(accuracy = 1),
    trans = "log10",
    oob = scales::squish,
    guide = guide_lancet_colourbar(barwidth = unit(5.0, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
  ) +
  labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "All infections\nper 100 000/year") +
  theme_lancet_heatmap(base_size = journal_compact_text_size)

extended12 <- ((p_ed12a | p_ed12b | p_ed12d) / (p_ed12e | p_ed12c | p_ed12f)) +
  plot_layout(guides = "keep", widths = c(1, 1, 1), heights = c(1, 1)) +
  plot_annotation(tag_levels = "A") &
  (theme(plot.margin = margin(4, 4, 4, 4)) + theme_lancet_tags())

save_appendix_figure(
  extended12,
  "extended_data_figure_12_fitness_resistance_grid",
  width = lancet_double_width * 1.35,
  height = 7.6
)
