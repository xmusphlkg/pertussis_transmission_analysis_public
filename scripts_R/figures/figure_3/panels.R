## Figure 3 panels -------------------------------------------------------------

plot_figure_3_panel_a <- function(data) {
  ggplot(data$endpoint_effect_matrix, aes(endpoint, strategy_label_plot, fill = median_relative_case_reduction)) +
    geom_tile(colour = "white", linewidth = lancet_heatmap_tile_linewidth) +
    geom_vline(xintercept = c(3.5, 5.5), colour = "white", linewidth = 1.05) +
    geom_text(
      aes(label = effect_label, colour = effect_text_colour),
      size = journal_heatmap_cell_text_size,
      lineheight = 0.82
    ) +
    scale_x_discrete(labels = data$endpoint_axis_labels) +
    scale_fill_gradientn(
      colours = c(manuscript_colour("light_grey"), manuscript_colour("sky"), manuscript_colour("blue")),
      limits = c(0, 0.50),
      breaks = seq(0, 0.50, by = 0.10),
      labels = label_lancet_percent(accuracy = 1),
      oob = scales::squish,
      name = "Median [cross-profile IQR] of reduction",
      guide = guide_lancet_colourbar(
        barwidth = unit(0.30, "cm"),
        barheight = unit(5.5, "cm"),
        title.position = "left",
        title.hjust = 0
      )
    ) +
    scale_colour_identity() +
    labs(x = NULL, y = NULL, tag = "a") +
    theme_lancet_heatmap(
      base_size = journal_dense_text_size,
      plot_margin = margin(4, 5, 4, 4),
      x_size = journal_dense_text_size - 0.7,
      legend_position = "right",
      legend_direction = "vertical"
    ) +
    theme(
      legend.title = element_text(angle = 90, hjust = 0, vjust = 0.5, lineheight = 0.92),
      legend.text = element_text(lineheight = 0.95),
      legend.title.position = "left"
    )
}

plot_figure_3_panel_b <- function(data) {
  ggplot() +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_segment(
      data = data$endpoint_gap_summary,
      aes(x = min_gap_pp, xend = max_gap_pp, y = strategy_label_plot, yend = strategy_label_plot),
      linewidth = 0.26,
      colour = manuscript_colour("pale_grey")
    ) +
    geom_segment(
      data = data$endpoint_gap_summary,
      aes(
        x = q25_gap_pp,
        xend = q75_gap_pp,
        y = strategy_label_plot,
        yend = strategy_label_plot,
        linetype = "IQR"
      ),
      linewidth = 1.05,
      colour = manuscript_colour("grey"),
      alpha = 0.74
    ) +
    geom_point(
      data = data$endpoint_gap,
      aes(infant_minus_child_adolescent_gap_pp, strategy_label_plot, colour = strategy, shape = "Profiles"),
      alpha = 0.50,
      size = 1.15,
      position = position_jitter(height = 0.075, width = 0)
    ) +
    geom_point(
      data = data$endpoint_gap_summary,
      aes(median_gap_pp, strategy_label_plot, shape = "Median"),
      fill = manuscript_colour("black"),
      colour = "white",
      stroke = 0.20,
      size = 2.15
    ) +
    scale_x_continuous(
      breaks = seq(-10, 30, by = 10),
      labels = label_lancet_number(accuracy = 1)
    ) +
    scale_colour_manual(values = strategy_colours, guide = "none") +
    scale_shape_profile_median() +
    scale_linetype_iqr() +
    coord_cartesian(xlim = c(-12, 30), clip = "off") +
    labs(
      x = "Infant-case reduction minus\nall-<18-case reduction (percentage points)",
      y = NULL,
      tag = "b"
    ) +
    theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 4, 4, 5), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      key_height = unit(0.34, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 0, b = 0),
      spacing_y = unit(0, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_3_age_contribution <- function(data) {
  ggplot(data$age_contribution_summary, aes(median_cases_averted_per_100k_under18, strategy_label_plot, fill = age_contribution)) +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_col(
      width = 0.68,
      colour = "white",
      linewidth = 0.16,
      position = position_stack(reverse = TRUE),
      key_glyph = ggplot2::draw_key_point
    ) +
    scale_x_continuous(
      breaks = seq(0, data$age_contribution_x_upper, by = 10),
      labels = label_lancet_number(accuracy = 1)
    ) +
    scale_fill_manual(
      values = data$age_stratum_colours,
      labels = data$age_stratum_display_labels,
      name = "Age group",
      guide = guide_legend(
        ncol = 1,
        byrow = TRUE,
        title.position = "top",
        title.hjust = 0,
        keywidth = unit(0.36, "cm"),
        keyheight = unit(0.46, "cm"),
        override.aes = list(shape = 22, size = 3.2, colour = NA, stroke = 0)
      )
    ) +
    coord_cartesian(xlim = c(-1, data$age_contribution_x_upper), clip = "off") +
    labs(
      x = "Median annualised symptomatic cases averted\nper 100 000 people aged <18 years",
      y = NULL,
      tag = "c"
    ) +
    theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 5, 4, 4), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour)
    ) +
    theme_lancet_inside_legend(
      key_width = unit(0.36, "cm"),
      key_height = unit(0.46, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 3, b = 3),
      spacing_y = unit(8, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_3_panel_c <- function(data) {
  ggplot() +
    geom_vline(xintercept = 0, linewidth = 0.24, colour = manuscript_colour("pale_grey")) +
    geom_segment(
      data = data$booster_effect_profile %>% filter(!china_profile),
      aes(
        x = all_under18_symptomatic_case_reduction,
        xend = adolescent_case_reduction,
        y = country_label_booster,
        yend = country_label_booster
      ),
      linewidth = 0.34,
      colour = manuscript_colour("light_grey")
    ) +
    geom_segment(
      data = data$booster_effect_profile %>% filter(china_profile),
      aes(
        x = all_under18_symptomatic_case_reduction,
        xend = adolescent_case_reduction,
        y = country_label_booster,
        yend = country_label_booster
      ),
      linewidth = 0.55,
      colour = manuscript_colour("black")
    ) +
    geom_point(
      data = data$booster_effect_long %>% filter(!china_profile),
      aes(case_reduction, country_label_booster, shape = outcome_label),
      size = 1.85,
      stroke = 0.30,
      colour = manuscript_colour("mid_grey"),
      fill = "white",
      alpha = 0.86
    ) +
    geom_point(
      data = data$booster_effect_long %>% filter(china_profile),
      aes(case_reduction, country_label_booster, shape = outcome_label),
      size = 2.45,
      stroke = 0.35,
      colour = lancet_text_colour,
      fill = lancet_text_colour
    ) +
    geom_text(
      data = data$booster_china_labels,
      aes(
        x = case_reduction + label_nudge_x,
        y = country_label_booster,
        label = label,
        hjust = label_hjust
      ),
      size = journal_point_label_text_size,
      colour = lancet_text_colour
    ) +
    scale_x_continuous(labels = label_lancet_percent(accuracy = 1), breaks = data$booster_x_breaks) +
    scale_shape_manual(
      values = c("All <18 symptomatic cases" = 21, "Adolescent cases" = 24),
      labels = c("All <18 cases", "Adolescent cases"),
      name = "Outcome",
      guide = guide_legend(
        ncol = 1,
        byrow = TRUE,
        title.position = "top",
        title.hjust = 0,
        keywidth = unit(0.36, "cm"),
        keyheight = unit(0.46, "cm"),
        override.aes = list(fill = "white", colour = manuscript_colour("grey"), size = 2.2)
      )
    ) +
    coord_cartesian(xlim = data$booster_x_limits, clip = "off") +
    labs(
      x = "Reduction in symptomatic cases\nunder adolescent booster scale-up (%)",
      y = NULL,
      tag = "c"
    ) +
    theme_lancet_panel(base_size = journal_dense_text_size, plot_margin = margin(4, 8, 4, 5), show_y_grid = TRUE) +
    theme(
      panel.grid.major.x = element_line(linewidth = 0.16, colour = lancet_grid_light_colour),
      axis.ticks.y = element_line(linewidth = lancet_axis_linewidth, colour = lancet_text_colour)
    ) +
    theme_lancet_inside_legend(
      key_height = unit(0.46, "cm"),
      title_lineheight = 0.90,
      text_lineheight = 1.16,
      text_margin = margin(t = 3, b = 3),
      spacing_y = unit(8, "pt"),
      box_margin = margin(0, 0, 1, 0),
      legend_margin = margin(3, 0, 3, 0)
    )
}

plot_figure_3_panels <- function(data) {
  list(
    a = plot_figure_3_panel_a(data),
    b = plot_figure_3_panel_b(data),
    c = plot_figure_3_panel_c(data),
    age_contribution = plot_figure_3_age_contribution(data)
  )
}
