## Extended Data Figure 8 panels ----------------------------------------------

extended_data_figure_8_fitness_colours <- function() {
  c(
    "0.85" = manuscript_colour("blue"),
    "0.90" = manuscript_discrete_core[["deep_teal"]],
    "0.95" = manuscript_colour("mid_grey"),
    "1.00" = manuscript_colour("black"),
    "1.05" = manuscript_colour("orange"),
    "1.10" = manuscript_colour("vermillion")
  )
}

extended_data_figure_8_fitness_linetypes <- function() {
  c(
    "0.85" = "solid",
    "0.90" = "longdash",
    "0.95" = "dotted",
    "1.00" = "solid",
    "1.05" = "twodash",
    "1.10" = "dotdash"
  )
}

extended_data_figure_8_calendar_year_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(numeric())
  span <- diff(range(x))
  by <- if (span <= 3) 1 else 2
  seq(floor(min(x)), ceiling(max(x)), by = by)
}

extended_data_figure_8_country_linetypes <- function() {
  c(
    "Australia" = "dotdash",
    "China" = "solid",
    "Japan" = "dashed"
  )
}

extended_data_figure_8_country_shapes <- function() {
  c(
    "Australia" = 15,
    "China" = 16,
    "Japan" = 17
  )
}

plot_extended_data_figure_8_country_hindcast <- function(data, obs, country_name, x_label = "Calendar year") {
  country_data <- data %>% filter(country == country_name)
  country_obs <- obs %>% filter(country == country_name)

  if (nrow(country_data) == 0) {
    return(
      ggplot() +
        annotate("text", x = 0, y = 0, label = paste(country_name, "- no data"), size = 2.4) +
        theme_void()
    )
  }

  country_ts <- country_data %>%
    group_by(fitness_label, calendar_year) %>%
    summarise(
      resistant_fraction = mean(model_resistant_fraction, na.rm = TRUE),
      .groups = "drop"
    )

  ggplot(
    country_ts,
    aes(
      calendar_year,
      resistant_fraction,
      colour = fitness_label,
      linetype = fitness_label,
      group = fitness_label
    )
  ) +
    geom_line(linewidth = 0.52, alpha = 0.88, lineend = "round") +
    geom_line(
      data = country_ts %>% filter(fitness_label == "1.00"),
      linewidth = 0.95, colour = manuscript_colour("black"), linetype = "solid"
    ) +
    geom_pointrange(
      data = country_obs,
      aes(
        x = calendar_year, y = observed_fraction,
        ymin = lower, ymax = upper
      ),
      colour = manuscript_colour("vermillion"), size = 0.8, linewidth = 0.5,
      shape = 18, inherit.aes = FALSE
    ) +
    scale_x_continuous(
      breaks = extended_data_figure_8_calendar_year_breaks,
      labels = function(x) as.character(as.integer(x))
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      limits = c(0, 1.02)
    ) +
    scale_colour_manual(
      values = extended_data_figure_8_fitness_colours(),
      drop = FALSE,
      name = expression(italic(f)[R])
    ) +
    scale_linetype_manual(
      values = extended_data_figure_8_fitness_linetypes(),
      drop = FALSE,
      name = expression(italic(f)[R])
    ) +
    labs(
      x = x_label,
      y = "Resistant fraction"
    ) +
    guides(
      colour = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = 0.8)),
      linetype = guide_legend(nrow = 1, byrow = TRUE)
    ) +
    theme_lancet() +
    theme(
      legend.position = "right",
      legend.key.size = unit(0.25, "cm")
    )
}

plot_extended_data_figure_8_panel_a <- function(data) {
  plot_extended_data_figure_8_country_hindcast(
    data$hindcast,
    data$observed_data,
    "Australia",
    x_label = "Calendar year"
  )
}

plot_extended_data_figure_8_panel_b <- function(data) {
  plot_extended_data_figure_8_country_hindcast(
    data$hindcast,
    data$observed_data,
    "China",
    x_label = "Calendar year"
  )
}

plot_extended_data_figure_8_panel_c <- function(data) {
  plot_extended_data_figure_8_country_hindcast(
    data$hindcast,
    data$observed_data,
    "Japan",
    x_label = "Calendar year"
  )
}

plot_extended_data_figure_8_panel_d <- function(data) {
  ggplot(
    data$scores,
    aes(
      fitness_R,
      mean_absolute_error,
      colour = country_label,
      linetype = country_label,
      shape = country_label
    )
  ) +
    geom_line(linewidth = 0.5, alpha = 0.7) +
    geom_point(size = 1.5, alpha = 0.8) +
    geom_point(
      data = data$best_fitness,
      aes(fitness_R, mean_absolute_error),
      shape = 18, size = 3.5, colour = "black"
    ) +
    ggrepel::geom_text_repel(
      data = data$best_fitness,
      aes(
        fitness_R, mean_absolute_error,
        label = paste0(country, "\n(f_R=", fitness_label, ")")
      ),
      size = 2.0, colour = "black",
      nudge_y = 0.02, segment.size = 0.2
    ) +
    scale_colour_manual(values = manuscript_country_focus_colours[c("Australia", "China", "Japan")]) +
    scale_linetype_manual(values = extended_data_figure_8_country_linetypes()) +
    scale_shape_manual(values = extended_data_figure_8_country_shapes()) +
    scale_x_continuous(breaks = seq(0.85, 1.10, 0.05)) +
    labs(
      x = expression(italic(f)[R]),
      y = "Mean absolute error\n(modelled vs observed)",
      colour = NULL,
      linetype = NULL,
      shape = NULL
    ) +
    theme_lancet() +
    theme(legend.position = "bottom")
}

plot_extended_data_figure_8_panels <- function(data) {
  list(
    a = plot_extended_data_figure_8_panel_a(data),
    b = plot_extended_data_figure_8_panel_b(data),
    c = plot_extended_data_figure_8_panel_c(data),
    d = plot_extended_data_figure_8_panel_d(data)
  )
}
