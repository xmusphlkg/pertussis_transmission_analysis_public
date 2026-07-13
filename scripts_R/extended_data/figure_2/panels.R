## Extended Data Figure 2 panels ----------------------------------------------

extended_data_figure_2_calibration_fingerprint_theme <- function() {
  theme_lancet_panel(
    base_size = journal_compact_text_size,
    plot_margin = margin(5, 4, 4, 4),
    show_x_grid = TRUE,
    show_y_grid = TRUE
  ) +
    theme(
      axis.ticks.y = element_blank(),
      legend.position = "none",
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    )
}

extended_data_figure_2_age_pattern_status_colours <- function() {
  c(
    "Pass" = manuscript_colour("green"),
    "Below threshold" = manuscript_colour("vermillion")
  )
}

plot_extended_data_figure_2_panel_a <- function(data) {
  data$observed_annual %>%
    ggplot(aes(Year, observed_reported_incidence)) +
    geom_line(linewidth = 0.28, colour = manuscript_colour("grey")) +
    facet_wrap(~country_code, scales = "free_y", nrow = 2) +
    scale_x_continuous(breaks = pretty_breaks(n = 2)) +
    scale_y_continuous(labels = label_number(accuracy = 0.1)) +
    labs(x = NULL, y = "Observed reported\nincidence per\n100 000/year", tag = "A") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.2))
}

plot_extended_data_figure_2_panel_b <- function(data) {
  data$observed %>%
    group_by(config_key, Year) %>%
    summarise(observed_cases = sum(Cases, na.rm = TRUE), .groups = "drop") %>%
    filter(config_key %in% data$calibration_diagnostic$country) %>%
    mutate(country_code = factor(country_codes[config_key], levels = country_codes[country_levels])) %>%
    ggplot(aes(Year, observed_cases)) +
    geom_line(linewidth = 0.35, colour = manuscript_colour("grey")) +
    geom_rect(
      data = data$calibration_diagnostic,
      aes(
        xmin = -Inf,
        xmax = Inf,
        ymin = fitted_temporal_range_low,
        ymax = fitted_temporal_range_high
      ),
      fill = manuscript_colour("vermillion"),
      alpha = 0.12,
      inherit.aes = FALSE
    ) +
    geom_hline(
      data = data$calibration_diagnostic,
      aes(yintercept = fitted_temporal_mean_reported_cases),
      linewidth = 0.4,
      colour = manuscript_colour("vermillion"),
      inherit.aes = FALSE
    ) +
    facet_wrap(~country_code, scales = "free_y", nrow = 2) +
    scale_x_continuous(breaks = pretty_breaks(n = 2)) +
    labs(x = NULL, y = "Annual reported cases\n(fitted range shaded)", tag = "B") +
    theme_lancet() +
    theme(axis.text.x = element_text(size = 5.2))
}

plot_extended_data_figure_2_panel_c <- function(data) {
  theme_calibration_fingerprint <- extended_data_figure_2_calibration_fingerprint_theme()

  ggplot(data$calibration_fingerprint, aes(model_observed_ratio, country_label_factor)) +
    annotate(
      "rect",
      xmin = data$calibration_thresholds$mean_ratio_min[[1]],
      xmax = data$calibration_thresholds$mean_ratio_max[[1]],
      ymin = -Inf,
      ymax = Inf,
      fill = manuscript_colour("pale_grey"),
      alpha = 0.38
    ) +
    geom_vline(xintercept = 1, linewidth = 0.26, linetype = "dashed", colour = manuscript_colour("black")) +
    geom_point(aes(fill = who_region), shape = 21, size = 1.95, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
    scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
    scale_x_continuous(
      breaks = c(0.75, 1.0, 1.25),
      labels = label_lancet_number(accuracy = 0.01),
      expand = expansion(mult = c(0.03, 0.03))
    ) +
    coord_cartesian(xlim = c(0.75, 1.25)) +
    labs(x = "Mean reported-\nincidence ratio", y = NULL, tag = "C") +
    theme_calibration_fingerprint
}

plot_extended_data_figure_2_panel_d <- function(data) {
  theme_calibration_fingerprint <- extended_data_figure_2_calibration_fingerprint_theme()

  ggplot(data$calibration_fingerprint, aes(calibration_interval_smape, country_label_factor)) +
    geom_vline(
      xintercept = data$calibration_thresholds$max_interval_smape[[1]],
      linewidth = 0.26,
      linetype = "dashed",
      colour = manuscript_colour("mid_grey")
    ) +
    geom_point(aes(fill = who_region), shape = 21, size = 1.95, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
    scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
    scale_x_continuous(
      breaks = c(0.5, 1.0, 1.3),
      labels = label_lancet_number(accuracy = 0.01),
      expand = expansion(mult = c(0.05, 0.07))
    ) +
    coord_cartesian(xlim = c(0.5, 1.35)) +
    labs(x = "Interval\nsMAPE", y = NULL, tag = "D") +
    theme_calibration_fingerprint +
    theme(axis.text.y = element_blank())
}

plot_extended_data_figure_2_panel_e <- function(data) {
  theme_calibration_fingerprint <- extended_data_figure_2_calibration_fingerprint_theme()

  ggplot(data$calibration_fingerprint, aes(calibration_peak_ratio, country_label_factor)) +
    annotate(
      "rect",
      xmin = data$calibration_thresholds$peak_ratio_min[[1]],
      xmax = data$calibration_thresholds$peak_ratio_max[[1]],
      ymin = -Inf,
      ymax = Inf,
      fill = manuscript_colour("pale_grey"),
      alpha = 0.38
    ) +
    geom_vline(xintercept = 1, linewidth = 0.26, linetype = "dashed", colour = manuscript_colour("black")) +
    geom_point(aes(fill = who_region), shape = 21, size = 1.95, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.88) +
    scale_fill_manual(values = c(region_colours, "Other" = manuscript_colour("mid_grey")), guide = "none") +
    scale_x_log10(
      breaks = c(0.25, 0.5, 1, 2, 4),
      labels = c("0·25", "0·5", "1", "2", "4"),
      expand = expansion(mult = c(0.04, 0.05))
    ) +
    coord_cartesian(xlim = c(0.25, 4)) +
    labs(x = "Peak ratio\n(log)", y = NULL, tag = "E") +
    theme_calibration_fingerprint +
    theme(axis.text.y = element_blank())
}

plot_extended_data_figure_2_panel_f <- function(data) {
  age_pattern_status_colours <- extended_data_figure_2_age_pattern_status_colours()

  ggplot(data$age_pattern_checks, aes(country_age_pattern_weight, country_label_factor)) +
    annotate(
      "rect",
      xmin = data$age_pattern_threshold,
      xmax = Inf,
      ymin = -Inf,
      ymax = Inf,
      fill = manuscript_colour("pale_green"),
      alpha = 0.28
    ) +
    geom_vline(xintercept = data$age_pattern_threshold, linewidth = 0.26, linetype = "dashed", colour = manuscript_colour("black")) +
    geom_segment(
      aes(x = 0, xend = country_age_pattern_weight, yend = country_label_factor),
      linewidth = 0.32,
      colour = manuscript_colour("pale_grey"),
      lineend = "round"
    ) +
    geom_point(aes(fill = age_pattern_status), shape = 21, size = 1.95, colour = manuscript_colour("black"), stroke = 0.22, alpha = 0.9) +
    geom_text(
      aes(label = age_pattern_status, colour = age_pattern_status),
      x = 0.98,
      hjust = 1,
      size = journal_point_label_text_size,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_fill_manual(values = age_pattern_status_colours, guide = "none") +
    scale_colour_manual(values = age_pattern_status_colours, guide = "none") +
    scale_x_continuous(
      breaks = c(0, data$age_pattern_threshold, 1),
      labels = label_lancet_number(accuracy = 0.1),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    scale_y_discrete(drop = FALSE) +
    coord_cartesian(xlim = c(0, 1), clip = "off") +
    labs(x = "Age-pattern\nweight", y = NULL, tag = "F") +
    theme_lancet_panel(
      base_size = journal_compact_text_size,
      plot_margin = margin(5, 4, 4, 4),
      show_x_grid = TRUE,
      show_y_grid = TRUE
    ) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      legend.position = "none",
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    )
}

plot_extended_data_figure_2_panel_g <- function(data) {
  ed2c_dodge <- position_dodge(width = 0.24)

  data$reporting_summary %>%
    select(scenario_label, annualized_reported_cases_per_100k,
           annualized_infections_per_100k, annualized_infant_cases_per_100k) %>%
    pivot_longer(-scenario_label, names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric_labels[metric], levels = c("All infections", "Reported cases", "Infant cases"))) %>%
    group_by(scenario_label, metric) %>%
    summarise(
      median = median(value, na.rm = TRUE),
      q025 = interval_quantile(value, 0.025),
      q975 = interval_quantile(value, 0.975),
      q25 = interval_quantile(value, 0.25),
      q75 = interval_quantile(value, 0.75),
      .groups = "drop"
    ) %>%
    ggplot(aes(scenario_label, median, colour = metric, group = metric)) +
    geom_errorbar(aes(ymin = q025, ymax = q975), width = 0.12, linewidth = 0.22, alpha = 0.55,
                  position = ed2c_dodge) +
    geom_errorbar(aes(ymin = q25, ymax = q75), width = 0, linewidth = 0.55,
                  position = ed2c_dodge) +
    geom_line(linewidth = 0.3, position = ed2c_dodge) +
    geom_point(size = 1.8, position = ed2c_dodge) +
    scale_y_log10(labels = label_number(accuracy = 1)) +
    scale_colour_manual(values = manuscript_metric_colours[c("All infections", "Reported cases", "Infant cases")]) +
    labs(x = NULL, y = "Median incidence\nper 100 000/year\n(log; country-profile ranges)", colour = NULL, tag = "G") +
    theme_lancet() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
}

plot_extended_data_figure_2_panel_h <- function(data) {
  data$reporting_long %>%
    ggplot(aes(age_group, country_label, fill = reporting_rate)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_fraction(
      limits = c(0, 0.7),
      breaks = seq(0, 0.7, by = 0.1),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      guide = guide_lancet_colourbar(
        barwidth = grid::unit(5.6, "cm"),
        barheight = grid::unit(0.24, "cm"),
        title.position = "top"
      )
    ) +
    labs(x = "Age group", y = NULL, fill = "Fitted reporting probability", tag = "H") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
}

plot_extended_data_figure_2_panels <- function(data) {
  list(
    a = plot_extended_data_figure_2_panel_a(data),
    b = plot_extended_data_figure_2_panel_b(data),
    c = plot_extended_data_figure_2_panel_c(data),
    d = plot_extended_data_figure_2_panel_d(data),
    e = plot_extended_data_figure_2_panel_e(data),
    f = plot_extended_data_figure_2_panel_f(data),
    g = plot_extended_data_figure_2_panel_g(data),
    h = plot_extended_data_figure_2_panel_h(data)
  )
}
