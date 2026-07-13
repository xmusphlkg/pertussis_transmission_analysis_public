## Extended Data Figure 11 panels ---------------------------------------------

extended_data_figure_11_panel_theme <- function() {
  theme_lancet_panel(base_size = journal_compact_text_size, plot_margin = margin(3, 3, 3, 3))
}

plot_extended_data_figure_11_panel_a <- function(data) {
  ggplot(data$portfolio, aes(median_relative_reduction_infant_cases, portfolio_label, colour = resistance_layer)) +
    geom_vline(xintercept = 0, linewidth = 0.25, colour = manuscript_colour("mid_grey")) +
    geom_errorbar(aes(xmin = lower, xmax = upper), width = 0.18, linewidth = 0.34, orientation = "y") +
    geom_point(aes(size = implementation_intensity), fill = "white") +
    geom_text(aes(label = first_label), nudge_x = 0.035, hjust = 0, size = 1.8, show.legend = FALSE) +
    scale_x_continuous(labels = pct_label, limits = c(0, 1.18), expand = expansion(mult = c(0.02, 0.08))) +
    scale_colour_manual(values = manuscript_resistance_layer_colours, name = NULL) +
    scale_size_continuous(range = c(1.2, 2.4), breaks = c(0, 3, 6), name = "Implementation\nintensity") +
    labs(x = "Median infant-case reduction vs current", y = NULL) +
    extended_data_figure_11_panel_theme() +
    theme(legend.position = "none")
}

plot_extended_data_figure_11_panel_b <- function(data) {
  ggplot(data$contact, aes(infant_contact_multiplier, median_infant_cases_per_100k, colour = strategy_label)) +
    geom_line(linewidth = 0.35) +
    geom_point(size = 1.6) +
    scale_x_continuous(breaks = c(0.75, 1, 1.25, 1.5)) +
    scale_y_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
    scale_colour_manual(values = manuscript_programme_colours[c("Current", "Infant-exposure strategy")], name = NULL) +
    labs(x = "Infant-contact multiplier", y = "Median infant cases per 100k/y") +
    extended_data_figure_11_panel_theme()
}

plot_extended_data_figure_11_panel_c <- function(data) {
  ggplot(data$maternal, aes(maternal_protection_duration_days, median_infant_case_reduction_vs_current_5y, colour = strategy_label)) +
    geom_line(linewidth = 0.35) +
    geom_point(size = 1.6) +
    scale_x_continuous(breaks = c(90, 180, 270)) +
    scale_y_continuous(labels = pct_label, limits = c(0, 0.70), expand = expansion(mult = c(0.02, 0.08))) +
    scale_colour_manual(
      values = manuscript_programme_colours[c("Direct antibody only", "Infant-exposure strategy")],
      name = NULL,
      guide = guide_legend(nrow = 2)
    ) +
    labs(x = "Maternal-protection duration, days", y = "Median infant-case reduction, 5 y") +
    extended_data_figure_11_panel_theme()
}

plot_extended_data_figure_11_panel_d <- function(data) {
  ggplot(data$temporal, aes(median_infant_cases_per_100k_5y, scenario_label, fill = temporal_dimension)) +
    geom_col(width = 0.65, colour = "black", linewidth = 0.15) +
    scale_x_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
    scale_fill_manual(values = manuscript_temporal_colours, name = NULL) +
    labs(x = "Median infant cases per 100k/y, 2027–2031", y = NULL) +
    extended_data_figure_11_panel_theme()
}

plot_extended_data_figure_11_panel_e <- function(data) {
  ggplot(data$event_scale, aes(annual_infant_cases_count + 0.5, scenario_label, colour = low_event)) +
    geom_point(position = position_jitter(height = 0.12, width = 0), size = 1.35, alpha = 0.85) +
    scale_x_log10(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.03, 0.08))) +
    scale_colour_manual(
      values = c(`FALSE` = manuscript_colour("mid_grey"), `TRUE` = manuscript_colour("vermillion")),
      labels = c("Other cells", "Low infant-event cells"),
      name = NULL
    ) +
    labs(x = "Annual infant cases, count + 0.5", y = NULL) +
    extended_data_figure_11_panel_theme()
}

plot_extended_data_figure_11_panel_f <- function(data) {
  ggplot(data$stochastic, aes(outbreak_probability_20plus, extinction_probability_3_or_fewer, colour = scenario_label)) +
    geom_point(aes(size = mean_household_clusters_touched), alpha = 0.86) +
    geom_text(aes(label = iso3), nudge_y = 0.018, size = 1.7, show.legend = FALSE) +
    scale_x_continuous(labels = pct_label, limits = c(0, 0.32), expand = expansion(mult = c(0.03, 0.08))) +
    scale_y_continuous(labels = pct_label, limits = c(0.52, 0.86), expand = expansion(mult = c(0.03, 0.08))) +
    scale_colour_manual(values = manuscript_stochastic_colours, name = NULL, guide = guide_legend(nrow = 2)) +
    scale_size_continuous(range = c(1.0, 2.5), name = "Mean household\nclusters", guide = guide_legend(nrow = 1)) +
    labs(x = "Pr(outbreak >=20 infections)", y = "Pr(extinction <=3 infections)") +
    extended_data_figure_11_panel_theme() +
    theme(legend.text = element_text(size = 5.4))
}

plot_extended_data_figure_11_panels <- function(data) {
  list(
    a = plot_extended_data_figure_11_panel_a(data),
    b = plot_extended_data_figure_11_panel_b(data),
    c = plot_extended_data_figure_11_panel_c(data),
    d = plot_extended_data_figure_11_panel_d(data),
    e = plot_extended_data_figure_11_panel_e(data),
    f = plot_extended_data_figure_11_panel_f(data)
  )
}
