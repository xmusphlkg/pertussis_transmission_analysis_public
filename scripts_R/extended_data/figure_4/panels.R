## Extended Data Figure 4 panels ----------------------------------------------

plot_extended_data_figure_4_timeseries_panel_a <- function(data) {
  data$weekly_country %>%
    ggplot(aes(simulation_year, infection_incidence)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["All infections"]) +
    facet_wrap(~country_code, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "All infections\nper 100 000/year") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.4))
}

plot_extended_data_figure_4_timeseries_panel_b <- function(data) {
  data$weekly_country %>%
    ggplot(aes(simulation_year, infant_case_incidence)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["Infant cases"]) +
    facet_wrap(~country_code, scales = "free_y", nrow = 1) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = label_number(accuracy = 1)) +
    labs(x = "Simulation year", y = "Infant cases\nper 100 000 infants/year") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.4))
}

plot_extended_data_figure_4_timeseries_panel_c <- function(data) {
  data$weekly_country %>%
    ggplot(aes(simulation_year, resistant_fraction)) +
    geom_line(linewidth = 0.25, colour = manuscript_metric_colours["Resistant"]) +
    facet_wrap(~country_code, ncol = 4) +
    scale_x_continuous(breaks = seq(0, 30, by = 10)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = "Simulation year", y = "Resistant infection\nfraction") +
    theme_lancet()
}

plot_extended_data_figure_4_timeseries_panel_d <- function(data) {
  data$age_strain_contribution %>%
    ggplot(aes(age_group, country_label, fill = country_share)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    facet_wrap(~strain_label, nrow = 1) +
    scale_fill_infant_burden(
      labels = label_lancet_percent(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(40, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = "Age group", y = NULL, fill = "Share of all infections") +
    theme_lancet_heatmap(x_angle = 90, x_hjust = 1, x_vjust = 0.5)
}

plot_extended_data_figure_4_fallback_panel_a <- function(data) {
  data$baseline_snapshot %>%
    ggplot(aes(annualized_infections_per_100k, country_burden_order)) +
    geom_point(size = 1.8, colour = manuscript_metric_colours["All infections"]) +
    scale_x_log10(labels = label_number(accuracy = 1)) +
    labs(x = "All infections per\n100 000/year (log)", y = NULL) +
    theme_lancet()
}

plot_extended_data_figure_4_fallback_panel_b <- function(data) {
  data$baseline_snapshot %>%
    ggplot(aes(annualized_infant_cases_per_100k, country_burden_order)) +
    geom_point(size = 1.8, colour = manuscript_metric_colours["Infant cases"]) +
    scale_x_log10(labels = label_number(accuracy = 1)) +
    labs(x = "Infant cases per\n100 000 infants/year (log)", y = NULL) +
    theme_lancet()
}

plot_extended_data_figure_4_fallback_panel_c <- function(data) {
  data$baseline_snapshot %>%
    ggplot(aes(y = country_burden_order)) +
    geom_segment(
      aes(x = resistant_fraction_start, xend = resistant_fraction_end, yend = country_burden_order),
      linewidth = 0.45,
      colour = manuscript_metric_colours["Resistant"],
      alpha = 0.75
    ) +
    geom_point(
      aes(x = resistant_fraction_start),
      shape = 21,
      fill = "white",
      colour = manuscript_metric_colours["Resistant"],
      size = 1.5,
      stroke = 0.3
    ) +
    geom_point(aes(x = resistant_fraction_end), colour = manuscript_metric_colours["Resistant"], size = 1.8) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(xlim = c(0, 1)) +
    labs(x = "Resistant fraction (start to end)", y = NULL) +
    theme_lancet()
}

plot_extended_data_figure_4_fallback_panel_d <- function(data) {
  data$origin_share_fallback %>%
    ggplot(aes(origin, country_label, fill = share)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_infant_burden(
      labels = label_lancet_percent(accuracy = 1),
      guide = guide_colourbar(
        barwidth = grid::unit(40, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = "Infection source history", y = NULL, fill = "Share of all infections") +
    theme_lancet_heatmap(x_angle = 90, x_hjust = 1, x_vjust = 0.5)
}

plot_extended_data_figure_4_panel_a <- function(data) {
  if (isTRUE(data$has_timeseries)) {
    return(plot_extended_data_figure_4_timeseries_panel_a(data))
  }
  plot_extended_data_figure_4_fallback_panel_a(data)
}

plot_extended_data_figure_4_panel_b <- function(data) {
  if (isTRUE(data$has_timeseries)) {
    return(plot_extended_data_figure_4_timeseries_panel_b(data))
  }
  plot_extended_data_figure_4_fallback_panel_b(data)
}

plot_extended_data_figure_4_panel_c <- function(data) {
  if (isTRUE(data$has_timeseries)) {
    return(plot_extended_data_figure_4_timeseries_panel_c(data))
  }
  plot_extended_data_figure_4_fallback_panel_c(data)
}

plot_extended_data_figure_4_panel_d <- function(data) {
  if (isTRUE(data$has_timeseries)) {
    return(plot_extended_data_figure_4_timeseries_panel_d(data))
  }
  plot_extended_data_figure_4_fallback_panel_d(data)
}

plot_extended_data_figure_4_panels <- function(data) {
  list(
    a = plot_extended_data_figure_4_panel_a(data),
    b = plot_extended_data_figure_4_panel_b(data),
    c = plot_extended_data_figure_4_panel_c(data),
    d = plot_extended_data_figure_4_panel_d(data)
  )
}
