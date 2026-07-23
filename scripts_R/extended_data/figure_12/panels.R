## Extended Data Figure 12 panels ---------------------------------------------

extended_data_figure_12_surface_axes <- function(data) {
  list(
    scale_x_continuous(
      breaks = data$fitness_breaks,
      labels = label_lancet_number(accuracy = 0.01),
      expand = expansion(mult = 0)
    ),
    scale_y_continuous(
      breaks = data$ve_inf_breaks,
      labels = label_lancet_percent(accuracy = 1),
      expand = expansion(mult = 0)
    ),
    coord_cartesian(xlim = data$fitness_range, ylim = data$ve_inf_range, expand = FALSE)
  )
}

extended_data_figure_12_diagnostic_parameter_labels <- function() {
  c(
    reporting_multiplier = "'Rpt.'",
    infant_contact_multiplier = "atop('Infant','contact')",
    VE_inf = "VE[inf]",
    fitness_R = "italic(f)[R]",
    relative_infectiousness_asymptomatic = "atop('Asympt.','rel.')",
    infectious_duration_asymptomatic = "atop('Asympt.','dur.')",
    resistance_management_uptake = "atop('Resistance','uptake')",
    PEP_coverage_multiplier = "atop('PEP','reach')"
  )
}

extended_data_figure_12_diagnostic_parameter_labeller <- function(values) {
  labels <- extended_data_figure_12_diagnostic_parameter_labels()[as.character(values)]
  missing_labels <- is.na(labels)
  if (any(missing_labels)) {
    labels[missing_labels] <- vapply(as.character(values[missing_labels]), shQuote, character(1))
  }
  parse(text = unname(labels))
}

plot_extended_data_figure_12_panel_a <- function(data) {
  ggplot(data$surface_resistant, aes(grid_fitness_R, grid_VE_inf)) +
    geom_raster(aes(fill = median_resistant_end), interpolate = TRUE) +
    geom_contour(
      aes(z = median_resistant_end),
      breaks = seq(0.1, 0.9, by = 0.1),
      colour = "white",
      alpha = 0.55,
      linewidth = 0.14
    ) +
    geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
    extended_data_figure_12_surface_axes(data) +
    scale_fill_resistance_fraction(
      limits = c(0, 1),
      breaks = c(0, 0.25, 0.50, 0.75, 1.00),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      guide = guide_lancet_colourbar(barwidth = unit(4.2, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
    ) +
    labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "End resistant\nfraction") +
    theme_lancet_heatmap(base_size = journal_compact_text_size)
}

plot_extended_data_figure_12_panel_b <- function(data) {
  ggplot(data$surface_infant, aes(grid_fitness_R, grid_VE_inf)) +
    geom_raster(aes(fill = median_infant_cases), interpolate = TRUE) +
    geom_contour(
      aes(z = median_infant_cases),
      breaks = data$infant_case_contour_breaks,
      colour = "white",
      alpha = 0.55,
      linewidth = 0.14
    ) +
    geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
    extended_data_figure_12_surface_axes(data) +
    scale_fill_infant_burden(
      limits = c(1, 4000),
      breaks = data$infant_case_colourbar_breaks,
      labels = label_lancet_comma(accuracy = 1),
      trans = "log10",
      oob = scales::squish,
      guide = guide_lancet_colourbar(barwidth = unit(5.0, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
    ) +
    labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "Infant cases\nper 100 000/year") +
    theme_lancet_heatmap(base_size = journal_compact_text_size)
}

plot_extended_data_figure_12_panel_c <- function(data) {
  diagnostic_status_labels <- c(
    "PSA varied" = "PSA varied",
    "Grid override" = "Grid override",
    "Excluded or fixed" = "Excluded/fixed"
  )

  ggplot(data$diagnostic_status, aes(parameter, country_burden_order, fill = parameter_status)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_x_discrete(labels = extended_data_figure_12_diagnostic_parameter_labeller, expand = c(0, 0)) +
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
}

plot_extended_data_figure_12_panel_d <- function(data) {
  ggplot(data$grid_benefit_full, aes(fitness_label, country_burden_order, fill = relative_benefit)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_fill_fraction(
      limits = c(0, 1),
      breaks = c(0, 0.25, 0.50, 0.75, 1.00),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      guide = guide_lancet_colourbar(barwidth = unit(3.6, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
    ) +
    labs(
      x = expression(italic(f)[R]),
      y = NULL,
      fill = paste0("High VE_inf benefit\n(", lancet_percent(data$high_grid_veinf), " vs ", lancet_percent(data$low_grid_veinf), ")")
    ) +
    theme_lancet_heatmap(
      base_size = journal_compact_text_size,
      x_angle = 90,
      x_hjust = 1,
      x_vjust = 0.5,
      x_size = 5.2
    )
}

plot_extended_data_figure_12_panel_e <- function(data) {
  ggplot(data$psa_interval_data, aes(y = y_position, colour = fitness_group)) +
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
      breaks = seq_along(data$psa_country_levels),
      labels = data$psa_country_levels,
      expand = expansion(add = c(0.45, 0.45))
    ) +
    scale_colour_manual(
      values = data$fitness_group_colours,
      breaks = names(data$fitness_group_colours),
      guide = guide_legend(nrow = 1, keywidth = unit(0.28, "cm"), keyheight = unit(0.18, "cm"))
    ) +
    labs(
      x = paste0("High VE_inf benefit (", lancet_percent(data$high_grid_veinf), " vs ", lancet_percent(data$low_grid_veinf), ")"),
      y = NULL,
      colour = NULL
    ) +
    theme_lancet_panel(base_size = journal_compact_text_size, show_y_grid = TRUE) +
    theme(
      legend.position = "bottom",
      panel.grid.major.x = element_line(linewidth = lancet_grid_linewidth, colour = lancet_grid_light_colour),
      axis.text.y = element_text(size = 5.4)
    )
}

plot_extended_data_figure_12_panel_f <- function(data) {
  ggplot(data$surface_all, aes(grid_fitness_R, grid_VE_inf)) +
    geom_raster(aes(fill = median_all_infections), interpolate = TRUE) +
    geom_contour(
      aes(z = median_all_infections),
      breaks = data$all_infection_contour_breaks,
      colour = "white",
      alpha = 0.55,
      linewidth = 0.14
    ) +
    geom_vline(xintercept = 1.0, linewidth = 0.32, linetype = "dashed", colour = "white", alpha = 0.65) +
    extended_data_figure_12_surface_axes(data) +
    scale_fill_infant_burden(
      limits = data$all_infection_limits,
      breaks = data$all_infection_colourbar_breaks,
      labels = label_lancet_comma(accuracy = 1),
      trans = "log10",
      oob = scales::squish,
      guide = guide_lancet_colourbar(barwidth = unit(3.8, "cm"), barheight = unit(0.24, "cm"), title.position = "left")
    ) +
    labs(x = expression(italic(f)[R]), y = expression(VE[inf]), fill = "All infections\nper 100 000/year") +
    theme_lancet_heatmap(base_size = journal_compact_text_size)
}

plot_extended_data_figure_12_panels <- function(data) {
  list(
    a = plot_extended_data_figure_12_panel_a(data),
    b = plot_extended_data_figure_12_panel_b(data),
    c = plot_extended_data_figure_12_panel_c(data),
    d = plot_extended_data_figure_12_panel_d(data),
    e = plot_extended_data_figure_12_panel_e(data),
    f = plot_extended_data_figure_12_panel_f(data)
  )
}
