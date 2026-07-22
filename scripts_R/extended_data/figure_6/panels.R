## Extended Data Figure 6 panels ----------------------------------------------

plot_extended_data_figure_6_panel_a <- function(data) {
  data$realized_start_diagnostic %>%
    ggplot(aes(target_resistance, resistant_fraction_start, fill = scenario_label)) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linewidth = 0.25,
      linetype = "dashed",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_point(shape = 21, size = 2.1, stroke = 0.25, colour = "black", alpha = 0.92) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_fill_manual(values = manuscript_resistance_scenario_colours) +
    labs(
      x = "Target resistant fraction at analysis start",
      y = "Realized starting resistant fraction",
      fill = NULL
    ) +
    theme_lancet()
}

plot_extended_data_figure_6_panel_b <- function(data) {
  data$resistant_rate %>%
    ggplot(aes(scenario_label, country_label, fill = resistant_rate)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_y_discrete(
      limits = rev(main_figure_country_order(as.character(data$resistant_rate$country_label))),
      drop = FALSE
    ) +
    scale_fill_infant_burden(
      labels = label_lancet_number(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(42, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = NULL, y = NULL, fill = "Resistant infections per 100 000/year") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
}

plot_extended_data_figure_6_panel_c <- function(data) {
  ed8c_dodge <- position_dodge(width = 0.24)

  data$treatment_pep %>%
    ggplot(aes(scenario_label, median_rate, colour = metric, group = metric)) +
    geom_errorbar(
      aes(ymin = q025, ymax = q975),
      width = 0.12,
      linewidth = 0.22,
      alpha = 0.55,
      position = ed8c_dodge
    ) +
    geom_errorbar(
      aes(ymin = q25, ymax = q75),
      width = 0,
      linewidth = 0.55,
      position = ed8c_dodge
    ) +
    geom_line(linewidth = 0.3, position = ed8c_dodge) +
    geom_point(size = 1.8, position = ed8c_dodge) +
    scale_y_continuous(labels = label_number(accuracy = 0.1)) +
    scale_colour_manual(values = manuscript_metric_colours[c("Treated cases", "PEP-averted cases")]) +
    labs(
      x = NULL,
      y = "Median events\nper 100 000/year\n(country-profile ranges)",
      colour = NULL
    ) +
    theme_lancet() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

plot_extended_data_figure_6_panel_d <- function(data) {
  if (identical(data$resistance_ts_mode, "simulation")) {
    panel <- data$resistance_ts %>%
      ggplot(aes(simulation_year, infection_incidence, colour = strain_label)) +
      geom_line(linewidth = 0.3) +
      facet_wrap(~country_label, scales = "free_y", nrow = 1) +
      scale_x_continuous(breaks = seq(0, 30, by = 10)) +
      scale_y_continuous(labels = label_number(accuracy = 1)) +
      labs(x = "Simulation year", y = "Infections per 100 000/year", colour = NULL)
  } else {
    panel <- data$resistance_ts %>%
      ggplot(aes(strain_label, infection_incidence, colour = strain_label)) +
      geom_point(size = 2.0) +
      facet_wrap(~country_label, scales = "free_y", nrow = 1) +
      scale_y_log10(labels = label_number(accuracy = 1)) +
      labs(x = NULL, y = "Infections per 100 000/year (log)", colour = NULL)
  }

  panel +
    scale_colour_manual(values = manuscript_metric_colours[c("Sensitive", "Resistant")]) +
    theme_lancet()
}

plot_extended_data_figure_6_panels <- function(data) {
  list(
    a = plot_extended_data_figure_6_panel_a(data),
    b = plot_extended_data_figure_6_panel_b(data),
    c = plot_extended_data_figure_6_panel_c(data),
    d = plot_extended_data_figure_6_panel_d(data)
  )
}
