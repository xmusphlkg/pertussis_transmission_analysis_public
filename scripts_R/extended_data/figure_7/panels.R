## Extended Data Figure 7 panels ----------------------------------------------

plot_extended_data_figure_7_panel_a <- function(data) {
  ggplot(data$lever_matrix, aes(lever, scenario_label, fill = active)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_manual(
      values = c("TRUE" = manuscript_colour("blue"), "FALSE" = manuscript_colour("light_grey")),
      guide = "none"
    ) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    labs(x = NULL, y = NULL) +
    theme_lancet_heatmap(x_angle = 0, x_hjust = 0.5, x_size = 4.9)
}

plot_extended_data_figure_7_panel_b <- function(data) {
  ggplot(data$intervention_outcomes, aes(scenario_short, country_label, fill = value)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    facet_wrap(~metric, nrow = 2) +
    scale_y_discrete(
      limits = rev(main_figure_country_order(as.character(data$intervention_outcomes$country_label))),
      drop = FALSE
    ) +
    scale_fill_reduction(
      midpoint = 0,
      limits = c(-0.25, 0.90),
      breaks = seq(-0.2, 0.8, by = 0.2),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      na.value = manuscript_colour("light_grey"),
      guide = guide_lancet_colourbar(
        barwidth = grid::unit(6.2, "cm"),
        barheight = grid::unit(0.24, "cm"),
        title.position = "left"
      )
    ) +
    labs(x = "Intervention\nstrategy", y = NULL, fill = "Relative\nreduction") +
    theme_lancet_heatmap(x_angle = 45, x_hjust = 1, x_size = 4.3) +
    theme(
      legend.key.width = unit(6.2, "cm"),
      legend.key.height = unit(0.24, "cm")
    )
}

plot_extended_data_figure_7_panel_c <- function(data) {
  ggplot(data$maternal_decomp, aes(relative_reduction_infant_cases, component, colour = component)) +
    geom_vline(xintercept = 0, linewidth = 0.25, colour = manuscript_colour("pale_grey")) +
    geom_point(
      size = 1.3,
      alpha = 0.6,
      position = position_jitter(height = 0.1, width = 0)
    ) +
    geom_errorbar(
      data = data$maternal_decomp_agg,
      aes(xmin = q025, xmax = q975, y = component),
      width = 0.22,
      linewidth = 0.28,
      colour = manuscript_colour("mid_grey"),
      alpha = 0.55,
      inherit.aes = FALSE,
      orientation = "y"
    ) +
    geom_errorbar(
      data = data$maternal_decomp_agg,
      aes(xmin = q25, xmax = q75, y = component),
      width = 0,
      linewidth = 0.7,
      colour = manuscript_colour("mid_grey"),
      inherit.aes = FALSE,
      orientation = "y"
    ) +
    geom_point(
      data = data$maternal_decomp_agg,
      aes(x = median_reduction, y = component),
      shape = 18,
      size = 3.0,
      colour = "black",
      inherit.aes = FALSE
    ) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    scale_colour_manual(values = data$maternal_decomp_colours, guide = "none") +
    labs(x = "Relative reduction in infant cases vs current\n(country-profile ranges)", y = NULL) +
    theme_lancet() +
    theme(plot.margin = margin(3, 10, 3, 3))
}

plot_extended_data_figure_7_panels <- function(data) {
  list(
    a = plot_extended_data_figure_7_panel_a(data),
    b = plot_extended_data_figure_7_panel_b(data),
    c = plot_extended_data_figure_7_panel_c(data)
  )
}
