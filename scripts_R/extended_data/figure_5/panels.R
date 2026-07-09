## Extended Data Figure 5 panels ----------------------------------------------

plot_extended_data_figure_5_panel_a <- function(data) {
  data$vaccine_parameters %>%
    ggplot(aes(effect, scenario_label, fill = value)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_text(aes(label = lancet_percent(value, accuracy = 1)), size = 1.9) +
    scale_fill_fraction(labels = label_lancet_percent(accuracy = 1)) +
    labs(x = "Vaccine effect parameter", y = NULL, fill = "Value") +
    theme_lancet_heatmap()
}

plot_extended_data_figure_5_panel_b <- function(data) {
  data$vaccine_reduction_data %>%
    ggplot(aes(scenario_short, country_label, fill = value)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    facet_wrap(~metric, nrow = 1) +
    scale_fill_reduction(
      midpoint = 0,
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      na.value = manuscript_colour("light_grey"),
      guide = guide_lancet_colourbar(
        barwidth = grid::unit(5.8, "cm"),
        barheight = grid::unit(0.24, "cm"),
        title.position = "top"
      )
    ) +
    labs(x = NULL, y = NULL, fill = "Relative reduction") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
}

plot_extended_data_figure_5_panel_c <- function(data) {
  data$origin_share_data %>%
    ggplot(aes(origin, scenario_short, fill = median_share)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_text(aes(label = interval_text, colour = text_colour), size = 1.75, lineheight = 0.82) +
    scale_fill_infant_burden(
      labels = label_lancet_percent(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(40, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    scale_colour_identity(guide = "none") +
    labs(x = "Infection source history", y = NULL, fill = "Median infection share") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
}

plot_extended_data_figure_5_panel_d <- function(data) {
  if (isTRUE(data$has_vaccine_sim)) {
    p <- data$vaccine_ts %>%
      ggplot(aes(simulation_year, infant_case_incidence, colour = scenario_short)) +
      geom_line(linewidth = 0.3) +
      facet_wrap(~country_label, scales = "free_y", nrow = 1) +
      scale_x_continuous(breaks = seq(0, 30, by = 10)) +
      scale_y_continuous(labels = label_number(accuracy = 1)) +
      labs(x = "Simulation year", y = "Infant cases per 100 000 infants/year", colour = NULL)
  } else {
    p <- data$vaccine_ts %>%
      ggplot(aes(scenario_short, annualized_infant_cases_per_100k, colour = scenario_short, group = 1)) +
      geom_line(linewidth = 0.3, colour = manuscript_colour("grey")) +
      geom_point(size = 1.8) +
      facet_wrap(~country_label, scales = "free_y", nrow = 1) +
      scale_y_log10(labels = label_number(accuracy = 1)) +
      labs(x = NULL, y = "Annualized infant cases per 100 000 (log)", colour = NULL)
  }

  p <- p +
    scale_colour_manual(values = manuscript_vaccine_profile_colours, guide = guide_legend(nrow = 2, byrow = TRUE)) +
    theme_lancet()

  if (!isTRUE(data$has_vaccine_sim)) {
    p <- p + theme(axis.text.x = element_text(angle = 35, hjust = 1))
  }

  p
}

plot_extended_data_figure_5_panels <- function(data) {
  list(
    a = plot_extended_data_figure_5_panel_a(data),
    b = plot_extended_data_figure_5_panel_b(data),
    c = plot_extended_data_figure_5_panel_c(data),
    d = plot_extended_data_figure_5_panel_d(data)
  )
}
