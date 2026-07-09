## Extended Data Figure 1 panels ----------------------------------------------

plot_extended_data_figure_1_panel_a <- function(data) {
  ggplot(data$coverage_plot, aes(coverage, country_label, colour = programme, shape = programme)) +
    geom_point(position = position_dodge(width = 0.45), size = 1.8) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    coord_cartesian(xlim = c(0, 1)) +
    scale_colour_manual(values = manuscript_coverage_colours) +
    labs(x = "Coverage", y = NULL, colour = NULL, shape = NULL) +
    theme_lancet()
}

plot_extended_data_figure_1_panel_b <- function(data) {
  ggplot(data$profile_inputs, aes(y = country_label)) +
    geom_segment(
      aes(x = routine_first_shot_months, xend = routine_last_shot_months, yend = country_label),
      linewidth = 0.45,
      colour = manuscript_colour("grey")
    ) +
    geom_point(aes(x = routine_first_shot_months), shape = 21, size = 1.7, fill = "white", colour = manuscript_colour("grey"), stroke = 0.35) +
    geom_point(
      aes(x = routine_last_shot_months, size = routine_dose_count, fill = maternal_program),
      shape = 21,
      colour = "black",
      stroke = 0.25
    ) +
    scale_x_continuous(labels = label_number(accuracy = 1)) +
    coord_cartesian(xlim = c(0, 200)) +
    scale_size_continuous(range = c(1.6, 3.8), breaks = c(4, 5, 6), name = "Routine doses") +
    scale_fill_manual(
      values = c("TRUE" = manuscript_colour("green"), "FALSE" = manuscript_colour("pale_grey")),
      labels = c("No", "Yes"),
      name = "Maternal programme"
    ) +
    guides(
      size = guide_legend(order = 1, nrow = 1, title.position = "top"),
      fill = guide_legend(order = 2, nrow = 1, title.position = "top")
    ) +
    labs(x = "Age at first and last routine dose, months", y = NULL) +
    theme_lancet() +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.direction = "horizontal"
    )
}

plot_extended_data_figure_1_panel_c <- function(data) {
  ggplot(data$contacts_summary, aes(source_age_group, country_label, fill = total_contacts)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    scale_fill_fraction(
      labels = label_lancet_number(accuracy = 0.1),
      guide = guide_colourbar(
        barwidth = grid::unit(34, "mm"),
        barheight = grid::unit(3, "mm"),
        title.position = "top"
      )
    ) +
    labs(x = "Source age group", y = NULL, fill = "Contacts per day") +
    theme_lancet_heatmap(x_angle = 35, x_hjust = 1)
}

plot_extended_data_figure_1_panel_d <- function(data) {
  ggplot(data$resistance_plot, aes(year, resistant_fraction)) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.18, linewidth = 0.25, colour = manuscript_colour("grey")) +
    geom_point(aes(fill = evidence_group, size = sample_size_plot), shape = 21, stroke = 0.25, colour = "black") +
    facet_wrap(~country_code, ncol = 4) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    scale_x_continuous(breaks = pretty_breaks(n = 3)) +
    scale_fill_manual(values = manuscript_evidence_colours, name = "Evidence type") +
    scale_size_continuous(range = c(1.6, 3.4), breaks = c(0, 50, 200, 600), name = "Sample size") +
    labs(x = "Evidence year", y = "Macrolide-resistant fraction") +
    theme_lancet() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_extended_data_figure_1_panels <- function(data) {
  list(
    a = plot_extended_data_figure_1_panel_a(data),
    b = plot_extended_data_figure_1_panel_b(data),
    c = plot_extended_data_figure_1_panel_c(data),
    d = plot_extended_data_figure_1_panel_d(data)
  )
}
