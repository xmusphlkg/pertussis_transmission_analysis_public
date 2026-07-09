## Extended Data Figure 10 panels ---------------------------------------------

extended_data_figure_10_panel_theme <- function() {
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 3, 3, 3))
}

plot_extended_data_figure_10_panel_a <- function(data) {
  ggplot(data$mechanism, aes(median_resistance_metric, fct_inorder(policy_short))) +
    geom_col(width = 0.62, fill = manuscript_colour("blue"), colour = "black", linewidth = 0.16) +
    scale_x_log10(labels = label_number(accuracy = 0.1), expand = expansion(mult = c(0.02, 0.08))) +
    labs(x = "Median resistant infections per 100k/y", y = NULL) +
    extended_data_figure_10_panel_theme()
}

plot_extended_data_figure_10_panel_b <- function(data) {
  ggplot(data$mechanism, aes(end_resistant_fraction, fct_inorder(policy_short))) +
    geom_segment(aes(x = 0, xend = end_resistant_fraction, yend = fct_inorder(policy_short)),
      linewidth = 0.35, colour = manuscript_colour("mid_grey")
    ) +
    geom_point(size = 1.8, colour = manuscript_colour("vermillion")) +
    scale_x_continuous(labels = pct_label, limits = c(0, 1), expand = expansion(mult = c(0.02, 0.06))) +
    labs(x = "End resistant fraction", y = NULL) +
    extended_data_figure_10_panel_theme()
}

plot_extended_data_figure_10_panel_c <- function(data) {
  ggplot(data$implementation, aes(median_infant_case_reduction_vs_current, policy_short, colour = pep_status)) +
    geom_vline(xintercept = 0, linewidth = 0.25, colour = "grey45") +
    geom_errorbar(aes(xmin = lower, xmax = upper), width = 0.18, linewidth = 0.35, orientation = "y") +
    geom_point(aes(size = countries_with_positive_reduction), fill = "white") +
    scale_x_continuous(labels = pct_label, expand = expansion(mult = c(0.08, 0.08))) +
    scale_colour_manual(
      values = c("Baseline PEP" = manuscript_colour("mid_grey"), "PEP improved" = manuscript_colour("green")),
      name = NULL
    ) +
    scale_size_continuous(range = c(1.2, 2.2), breaks = c(0, 4, 8), name = "Countries\npositive") +
    guides(colour = guide_legend(nrow = 1, byrow = TRUE), size = guide_legend(nrow = 1)) +
    labs(x = "Median infant-case\nreduction vs current", y = NULL) +
    extended_data_figure_10_panel_theme() +
    theme(
      axis.text.y = element_text(size = journal_compact_text_size - 0.6, lineheight = 0.92),
      legend.text = element_text(size = 5.8),
      legend.key.size = unit(0.24, "cm")
    )
}

plot_extended_data_figure_10_panel_d <- function(data) {
  ggplot(data$preference, aes(resistance_weight_lambda, countries_preferred, colour = strategy_label)) +
    geom_line(linewidth = 0.38) +
    geom_point(
      data = filter(data$preference, resistance_weight_lambda %in% seq(0, 1, by = 0.20)),
      size = 1.4
    ) +
    scale_x_continuous(labels = pct_label, breaks = c(0, 0.5, 1), expand = expansion(mult = c(0.02, 0.04))) +
    scale_y_continuous(breaks = c(0, 5, 10), limits = c(0, 10), expand = expansion(mult = c(0.02, 0.06))) +
    scale_colour_manual(
      values = manuscript_programme_colours[c("Timeliness", "Infant exposure", "Adolescent", "Resistance mgmt")],
      name = NULL
    ) +
    labs(x = "Weight on resistant-infection reduction", y = "Countries preferred") +
    extended_data_figure_10_panel_theme()
}

plot_extended_data_figure_10_panel_e <- function(data) {
  ggplot(data$thresholds, aes(resistance_prevalence, median_minimum_VE_inf, colour = comparator)) +
    geom_line(linewidth = 0.35) +
    geom_point(size = 1.7) +
    geom_text(aes(label = countries_label), vjust = -0.75, size = 1.75, show.legend = FALSE) +
    scale_x_continuous(labels = pct_label, breaks = c(0, 0.5, 1), expand = expansion(mult = c(0.04, 0.08))) +
    scale_y_continuous(labels = pct_label, breaks = seq(0, 0.6, by = 0.2), expand = expansion(mult = c(0.02, 0.10))) +
    coord_cartesian(ylim = c(0, data$threshold_y_upper), clip = "off") +
    scale_colour_manual(
      values = c(
        "Resistance-guided care" = manuscript_colour("blue"),
        "Infant-exposure strategy" = manuscript_colour("green"),
        "25% infant-case reduction" = manuscript_colour("orange"),
        "50% infant-case reduction" = manuscript_colour("magenta"),
        "75% infant-case reduction" = manuscript_colour("vermillion")
      ),
      name = NULL
    ) +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
    labs(x = "Resistance prevalence in threshold grid", y = "Minimum VE_inf") +
    extended_data_figure_10_panel_theme() +
    theme(
      legend.text = element_text(size = journal_compact_text_size - 0.8),
      legend.key.size = unit(0.24, "cm")
    )
}

plot_extended_data_figure_10_panels <- function(data) {
  list(
    a = plot_extended_data_figure_10_panel_a(data),
    b = plot_extended_data_figure_10_panel_b(data),
    c = plot_extended_data_figure_10_panel_c(data),
    d = plot_extended_data_figure_10_panel_d(data),
    e = plot_extended_data_figure_10_panel_e(data)
  )
}
