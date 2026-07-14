## Extended Data Figure 9 panels ----------------------------------------------

extended_data_figure_9_panel_theme <- function() {
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 3, 3, 3))
}

extended_data_figure_9_age_pattern_programme_colours <- function() {
  c(
    "All profiles" = manuscript_colour("mid_grey"),
    "Profiles with age data" = manuscript_colour("sky"),
    "Age-pattern weighted" = manuscript_colour("blue")
  )
}

extended_data_figure_9_age_pattern_programme_x_limits <- function(data) {
  x_range <- range(
    c(
      0,
      data$weighted_median_primary_case_reduction,
      data$weighted_iqr_low_primary_case_reduction,
      data$weighted_iqr_high_primary_case_reduction
    ),
    na.rm = TRUE
  )

  c(
    floor((x_range[[1]] - 0.01) / 0.05) * 0.05,
    ceiling((x_range[[2]] + 0.01) / 0.05) * 0.05
  )
}

plot_extended_data_figure_9_panel_a <- function(data) {
  ggplot(data$routine, aes(strategy_label, median_relative_reduction_infant_cases, fill = timeliness_applied)) +
    geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey45") +
    geom_col(width = 0.68, colour = "black", linewidth = 0.18) +
    geom_text(aes(label = countries_with_positive_reduction), vjust = -0.35, size = journal_heatmap_cell_text_size) +
    scale_y_continuous(labels = pct_label, limits = c(-0.08, 0.42), expand = expansion(mult = c(0.02, 0.10))) +
    scale_fill_manual(values = manuscript_binary_colours, guide = "none") +
    labs(x = NULL, y = "Median infant-case reduction", caption = "Text: countries with positive reduction") +
    extended_data_figure_9_panel_theme() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

plot_extended_data_figure_9_panel_b <- function(data) {
  ggplot(data$horizon, aes(analysis_window, scenario_label, fill = median_rank)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_text(aes(label = sprintf("%.0f", median_rank)), size = journal_heatmap_cell_text_size, colour = "black") +
    scale_fill_rank(
      limits = c(1, 10),
      name = "Median rank",
      guide = guide_lancet_colourbar(barwidth = unit(4.2, "cm"))
    ) +
    labs(x = "Analysis window", y = NULL) +
    theme_lancet_heatmap(
      base_size = journal_compact_text_size,
      plot_margin = margin(3, 3, 3, 3),
      x_angle = 35,
      x_hjust = 1
    )
}

plot_extended_data_figure_9_panel_c <- function(data) {
  ggplot(data$age_summary, aes(age_group, scenario_label, fill = median_reduction)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_text(aes(label = pct_label(median_reduction)), size = journal_heatmap_cell_text_size) +
    scale_fill_reduction(
      midpoint = 0,
      breaks = c(0, 0.25, 0.50, 0.75),
      labels = pct_label,
      name = "Reduction",
      guide = guide_lancet_colourbar(barwidth = unit(2.6, "cm"))
    ) +
    labs(x = "Infant age group", y = NULL) +
    theme_lancet_heatmap(
      base_size = journal_compact_text_size,
      plot_margin = margin(3, 3, 3, 3)
    )
}

plot_extended_data_figure_9_panel_d <- function(data) {
  ggplot(data$stability, aes(share, scenario_label, colour = diagnostic)) +
    geom_point(size = 1.7) +
    scale_x_continuous(labels = pct_label, limits = c(0, 1), expand = expansion(mult = c(0.02, 0.05))) +
    scale_colour_manual(values = manuscript_diagnostic_colours[c("Window cells top 2", "Infant-age cells top 2")]) +
    labs(x = "Share of diagnostic cells", y = NULL, colour = NULL) +
    extended_data_figure_9_panel_theme()
}

plot_extended_data_figure_9_panel_e <- function(data) {
  ggplot(data$psa, aes(frequency_rank_1, strategy_label)) +
    geom_col(width = 0.65, fill = manuscript_diagnostic_colours["Top 2"], colour = "black", linewidth = 0.15) +
    geom_point(aes(x = frequency_top_2), size = 1.5, colour = manuscript_diagnostic_colours["Within 10% of best"]) +
    scale_x_continuous(labels = pct_label, limits = c(0, 1), expand = expansion(mult = c(0.01, 0.04))) +
    labs(
      x = "Selected-parameter frequency",
      y = NULL,
      caption = "Primary <18 endpoint; bars: rank 1; dots: top 2"
    ) +
    extended_data_figure_9_panel_theme()
}

plot_extended_data_figure_9_panel_f <- function(data) {
  ggplot(data$age_pattern, aes(ordering_basis, scenario_class_label, fill = class_rank)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_text(aes(label = sprintf("%.0f", class_rank)), size = journal_heatmap_cell_text_size) +
    scale_fill_rank(
      limits = c(1, 6),
      name = "Class rank",
      guide = guide_lancet_colourbar(barwidth = unit(2.6, "cm"))
    ) +
    labs(x = "Ordering basis", y = NULL) +
    theme_lancet_heatmap(
      base_size = journal_compact_text_size,
      plot_margin = margin(3, 3, 3, 3),
      x_angle = 25,
      x_hjust = 1
    )
}

plot_extended_data_figure_9_panel_g <- function(data) {
  x_limits <- extended_data_figure_9_age_pattern_programme_x_limits(data$age_pattern_programme)

  ggplot(
    data$age_pattern_programme,
    aes(weighted_median_primary_case_reduction, strategy_label_plot, fill = ordering_basis_label)
  ) +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_col(
      position = position_dodge(width = 0.74),
      width = 0.62,
      colour = "white",
      linewidth = 0.12,
      alpha = 0.92
    ) +
    geom_errorbar(
      aes(
        xmin = weighted_iqr_low_primary_case_reduction,
        xmax = weighted_iqr_high_primary_case_reduction
      ),
      orientation = "y",
      position = position_dodge(width = 0.74),
      width = 0.18,
      linewidth = 0.28,
      colour = lancet_text_colour
    ) +
    scale_x_continuous(
      labels = pct_label,
      breaks = seq(x_limits[[1]], x_limits[[2]], by = 0.05)
    ) +
    scale_fill_manual(
      values = extended_data_figure_9_age_pattern_programme_colours(),
      labels = c("All", "Age data", "Weighted"),
      name = NULL,
      guide = guide_legend(nrow = 1, byrow = TRUE)
    ) +
    coord_cartesian(xlim = x_limits, clip = "off") +
    labs(x = "Median reduction in symptomatic cases among people aged <18 years", y = NULL) +
    theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 5, 3, 3), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      legend.position = "top",
      axis.ticks.y = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour)
    )
}

plot_extended_data_figure_9_panels <- function(data) {
  list(
    a = plot_extended_data_figure_9_panel_a(data),
    b = plot_extended_data_figure_9_panel_b(data),
    c = plot_extended_data_figure_9_panel_c(data),
    d = plot_extended_data_figure_9_panel_d(data),
    e = plot_extended_data_figure_9_panel_e(data),
    f = plot_extended_data_figure_9_panel_f(data),
    g = plot_extended_data_figure_9_panel_g(data)
  )
}
